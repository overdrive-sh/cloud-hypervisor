// Copyright © 2026 Overdrive contributors
//
// SPDX-License-Identifier: Apache-2.0
//

//! virtio-vsock device backed by the host kernel's `vhost-vsock` driver.
//!
//! Unlike [`super::Vsock`], which terminates guest vsock traffic in a
//! userspace muxer and proxies it to Unix domain sockets, this device hands
//! the RX and TX virtqueues to `/dev/vhost-vsock`. The kernel vhost worker
//! then moves packets directly between the guest rings and the host kernel's
//! `AF_VSOCK` stack: host applications reach the guest with ordinary
//! `AF_VSOCK` sockets addressed to the guest CID, and the VMM never touches
//! the payload.
//!
//! The VMM keeps ownership of the event virtqueue (the kernel driver does not
//! service it). That queue only carries `VIRTIO_VSOCK_EVENT_TRANSPORT_RESET`,
//! which a device emits after live migration; this device does not support
//! migration, so the queue is drained and left untouched.

use std::os::unix::io::AsRawFd;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier};
use std::{io, result};

use anyhow::anyhow;
use byteorder::{ByteOrder, LittleEndian};
use event_monitor::event;
use log::{error, info, warn};
use seccompiler::SeccompAction;
use thiserror::Error;
use vhost::vhost_kern::VhostKernBackend;
use vhost::vhost_kern::vsock::Vsock as VhostKernVsock;
use vhost::vsock::VhostVsock;
use vhost::{VhostBackend, VhostUserMemoryRegionInfo, VringConfigData};
use virtio_queue::QueueT;
use vm_memory::{Address, GuestAddressSpace, GuestMemory, GuestMemoryAtomic, GuestMemoryRegion};
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vm_virtio::AccessPlatform;
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};

use crate::device::ActivationContext;
use crate::seccomp_filters::Thread;
use crate::{
    ActivateError, ActivateResult, EPOLL_HELPER_EVENT_LAST, EpollHelper, EpollHelperError,
    EpollHelperHandler, GuestMemoryMmap, GuestRegionMmap, VIRTIO_F_RING_EVENT_IDX,
    VIRTIO_F_RING_INDIRECT_DESC, VIRTIO_F_VERSION_1, VirtioCommon, VirtioDevice, VirtioDeviceType,
    VirtioInterrupt, VirtioInterruptType,
};

const QUEUE_SIZE: u16 = 256;
const NUM_QUEUES: usize = 3;
const QUEUE_SIZES: &[u16] = &[QUEUE_SIZE; NUM_QUEUES];

/// RX and TX are serviced by the kernel; the event queue (index 2) stays in the VMM.
const KERNEL_QUEUES: usize = 2;
const EVENT_QUEUE_INDEX: usize = 2;

/// `VIRTIO_VSOCK_F_SEQPACKET`, defined in `include/uapi/linux/virtio_vsock.h`.
const VIRTIO_VSOCK_F_SEQPACKET: u32 = 1;

/// Features this device is prepared to forward between the guest driver and
/// the kernel backend. The kernel's own offer is intersected with this mask.
const SUPPORTED_FEATURES: u64 = (1u64 << VIRTIO_F_VERSION_1)
    | (1u64 << VIRTIO_F_RING_EVENT_IDX)
    | (1u64 << VIRTIO_F_RING_INDIRECT_DESC)
    | (1u64 << VIRTIO_VSOCK_F_SEQPACKET);

const EVENT_QUEUE_KICK: u16 = EPOLL_HELPER_EVENT_LAST + 1;
/// First of `KERNEL_QUEUES` consecutive ids used for interrupt forwarding.
const CALL_EVENT_BASE: u16 = EPOLL_HELPER_EVENT_LAST + 2;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to open /dev/vhost-vsock")]
    Open(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock owner")]
    SetOwner(#[source] vhost::Error),
    #[error("Failed to get vhost-vsock features")]
    GetFeatures(#[source] vhost::Error),
    #[error("Failed to assign guest CID {0} to vhost-vsock")]
    SetGuestCid(u64, #[source] vhost::Error),
    #[error("The vhost-vsock backend does not support VIRTIO_F_ACCESS_PLATFORM (IOMMU)")]
    AccessPlatformUnsupported,
    #[error("The vhost-vsock backend does not support snapshot/restore")]
    RestoreUnsupported,
    #[error("Failed to set vhost-vsock features")]
    SetFeatures(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock memory table")]
    SetMemTable(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock vring size")]
    SetVringNum(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock vring addresses")]
    SetVringAddr(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock vring base")]
    SetVringBase(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock vring call eventfd")]
    SetVringCall(#[source] vhost::Error),
    #[error("Failed to set vhost-vsock vring kick eventfd")]
    SetVringKick(#[source] vhost::Error),
    #[error("Failed to create interrupt forwarding eventfd")]
    CreateCallEventFd(#[source] io::Error),
    #[error("Failed to start vhost-vsock")]
    Start(#[source] vhost::Error),
    #[error("Failed to stop vhost-vsock")]
    Stop(#[source] vhost::Error),
    #[error("Driver activated {0} queues, vhost-vsock requires {NUM_QUEUES}")]
    QueueCount(usize),
}

pub type Result<T> = result::Result<T, Error>;

/// Worker for the VMM-owned part of the device: drains event-queue kicks and,
/// when the transport offers no irqfd for a kernel queue, forwards the kernel's
/// call eventfd to the guest interrupt. It never touches vsock payload.
struct VhostKernelVsockEpollHandler {
    event_queue_evt: EventFd,
    /// `(queue_index, call_evt)` for queues whose interrupt the kernel cannot
    /// raise directly.
    forwarded_calls: Vec<(u16, EventFd)>,
    interrupt_cb: Arc<dyn VirtioInterrupt>,
    kill_evt: EventFd,
    pause_evt: EventFd,
}

impl VhostKernelVsockEpollHandler {
    fn run(
        &mut self,
        paused: &AtomicBool,
        paused_sync: &Barrier,
    ) -> result::Result<(), EpollHelperError> {
        let mut helper = EpollHelper::new(&self.kill_evt, &self.pause_evt)?;
        helper.add_event(self.event_queue_evt.as_raw_fd(), EVENT_QUEUE_KICK)?;
        for (slot, (_, call_evt)) in self.forwarded_calls.iter().enumerate() {
            helper.add_event(call_evt.as_raw_fd(), CALL_EVENT_BASE + slot as u16)?;
        }
        helper.run(paused, paused_sync, self)?;
        Ok(())
    }
}

impl EpollHelperHandler for VhostKernelVsockEpollHandler {
    fn handle_event(
        &mut self,
        _helper: &mut EpollHelper,
        event: &epoll::Event,
    ) -> result::Result<(), EpollHelperError> {
        let ev_type = event.data as u16;
        if ev_type == EVENT_QUEUE_KICK {
            self.event_queue_evt.read().map_err(|e| {
                EpollHelperError::HandleEvent(anyhow!("Failed to read event queue kick: {e:?}"))
            })?;
            return Ok(());
        }

        let slot = ev_type
            .checked_sub(CALL_EVENT_BASE)
            .map(usize::from)
            .filter(|slot| *slot < self.forwarded_calls.len())
            .ok_or_else(|| {
                EpollHelperError::HandleEvent(anyhow!("Unknown event for vhost-kernel vsock"))
            })?;
        let (queue_index, call_evt) = &self.forwarded_calls[slot];
        call_evt.read().map_err(|e| {
            EpollHelperError::HandleEvent(anyhow!("Failed to read vhost call eventfd: {e:?}"))
        })?;
        self.interrupt_cb
            .trigger(VirtioInterruptType::Queue(*queue_index))
            .map_err(|e| {
                EpollHelperError::HandleEvent(anyhow!("Failed to signal used queue: {e:?}"))
            })
    }
}

/// Virtio vsock device whose data path is the host kernel `vhost-vsock` driver.
pub struct VhostKernelVsock {
    common: VirtioCommon,
    id: String,
    cid: u64,
    vhost: VhostKernVsock<GuestMemoryAtomic<GuestMemoryMmap>>,
    seccomp_action: SeccompAction,
    exit_evt: EventFd,
    /// True while the kernel backend is servicing the rings.
    running: bool,
}

impl VhostKernelVsock {
    /// Create the device, open `/dev/vhost-vsock` and claim `cid` in the host
    /// kernel. The claim is held until the device is dropped, so a CID that is
    /// already in use is reported here (typically `EADDRINUSE`) rather than when
    /// the guest driver activates the device.
    pub fn new(
        id: String,
        cid: u32,
        mem: GuestMemoryAtomic<GuestMemoryMmap>,
        access_platform_enabled: bool,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        restoring: bool,
    ) -> Result<Self> {
        if restoring {
            return Err(Error::RestoreUnsupported);
        }
        if access_platform_enabled {
            return Err(Error::AccessPlatformUnsupported);
        }

        let cid = u64::from(cid);
        let vhost = VhostKernVsock::new(mem).map_err(Error::Open)?;
        vhost.set_owner().map_err(Error::SetOwner)?;
        let kernel_features = vhost.get_features().map_err(Error::GetFeatures)?;
        vhost
            .set_guest_cid(cid)
            .map_err(|e| Error::SetGuestCid(cid, e))?;

        info!("Created vhost-kernel vsock {id} with guest CID {cid}");

        Ok(VhostKernelVsock {
            common: VirtioCommon {
                device_type: VirtioDeviceType::Vsock as u32,
                avail_features: kernel_features & SUPPORTED_FEATURES,
                acked_features: 0,
                paused_sync: Some(Arc::new(Barrier::new(2))),
                queue_sizes: QUEUE_SIZES.to_vec(),
                min_queues: NUM_QUEUES as u16,
                ..Default::default()
            },
            id,
            cid,
            vhost,
            seccomp_action,
            exit_evt,
            running: false,
        })
    }

    fn set_mem_table(&self) -> Result<()> {
        let mem = self.vhost.mem().memory();
        let regions = mem
            .iter()
            .map(|region| VhostUserMemoryRegionInfo {
                guest_phys_addr: region.start_addr().raw_value(),
                memory_size: region.len(),
                userspace_addr: region.as_ptr() as u64,
                mmap_offset: 0,
                mmap_handle: -1,
            })
            .collect::<Vec<_>>();
        self.vhost
            .set_mem_table(&regions)
            .map_err(Error::SetMemTable)
    }

    /// Program the kernel backend with the driver-negotiated rings and start it.
    /// Returns the call eventfds created for queues without an irqfd notifier.
    fn setup_backend(
        &mut self,
        queues: &[(usize, virtio_queue::Queue, EventFd)],
        interrupt_cb: &dyn VirtioInterrupt,
    ) -> Result<Vec<(u16, EventFd)>> {
        self.vhost
            .set_features(self.common.acked_features & SUPPORTED_FEATURES)
            .map_err(Error::SetFeatures)?;
        self.set_mem_table()?;

        let mut forwarded_calls = Vec::new();
        for (queue_index, queue, queue_evt) in
            queues.iter().filter(|(index, _, _)| *index < KERNEL_QUEUES)
        {
            self.vhost
                .set_vring_num(*queue_index, queue.size())
                .map_err(Error::SetVringNum)?;
            let config_data = VringConfigData {
                queue_max_size: queue.max_size(),
                queue_size: queue.size(),
                flags: 0,
                desc_table_addr: queue.desc_table(),
                used_ring_addr: queue.used_ring(),
                avail_ring_addr: queue.avail_ring(),
                log_addr: None,
            };
            self.vhost
                .set_vring_addr(*queue_index, &config_data)
                .map_err(Error::SetVringAddr)?;
            // A freshly activated ring has consumed no descriptors yet.
            self.vhost
                .set_vring_base(*queue_index, 0)
                .map_err(Error::SetVringBase)?;

            let queue_id = *queue_index as u16;
            if let Some(irqfd) = interrupt_cb.notifier(VirtioInterruptType::Queue(queue_id)) {
                self.vhost
                    .set_vring_call(*queue_index, &irqfd)
                    .map_err(Error::SetVringCall)?;
            } else {
                let call_evt = EventFd::new(EFD_NONBLOCK).map_err(Error::CreateCallEventFd)?;
                self.vhost
                    .set_vring_call(*queue_index, &call_evt)
                    .map_err(Error::SetVringCall)?;
                forwarded_calls.push((queue_id, call_evt));
            }

            self.vhost
                .set_vring_kick(*queue_index, queue_evt)
                .map_err(Error::SetVringKick)?;
        }

        self.vhost.start().map_err(Error::Start)?;
        self.running = true;
        Ok(forwarded_calls)
    }

    /// Stop the kernel backend from servicing the rings.
    fn stop_backend(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }
        self.vhost.stop().map_err(Error::Stop)?;
        self.running = false;
        Ok(())
    }

    fn activate_inner(&mut self, context: ActivationContext) -> ActivateResult {
        let ActivationContext {
            interrupt_cb,
            queues,
            device_status,
            ..
        } = context;

        if queues.len() != NUM_QUEUES {
            error!("{}", Error::QueueCount(queues.len()));
            return Err(ActivateError::BadActivate);
        }

        self.common.activate(&queues, interrupt_cb.clone())?;

        let forwarded_calls = match self.setup_backend(&queues, interrupt_cb.as_ref()) {
            Ok(calls) => calls,
            Err(e) => {
                // Leave the kernel backend stopped and the device inactive.
                let _ = self.stop_backend();
                self.common.reset();
                return Err(ActivateError::ActivateVhostKernelVsock(e));
            }
        };

        let (kill_evt, pause_evt) = self.common.dup_eventfds()?;
        let event_queue_evt = queues
            .iter()
            .find(|(index, _, _)| *index == EVENT_QUEUE_INDEX)
            .map(|(_, _, evt)| evt.try_clone())
            .ok_or(ActivateError::BadActivate)?
            .map_err(ActivateError::CloneEventFd)?;

        let mut handler = VhostKernelVsockEpollHandler {
            event_queue_evt,
            forwarded_calls,
            interrupt_cb: interrupt_cb.clone(),
            kill_evt,
            pause_evt,
        };
        let paused = self.common.paused.clone();
        let paused_sync = self.common.paused_sync.clone();

        if let Err(e) = self.common.spawn_worker(
            &self.id,
            &self.seccomp_action,
            Thread::VirtioVhostKernelVsock,
            &self.exit_evt,
            device_status,
            interrupt_cb,
            move || handler.run(&paused, paused_sync.as_ref().unwrap()),
        ) {
            let _ = self.stop_backend();
            return Err(e);
        }

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }
}

impl VirtioDevice for VhostKernelVsock {
    fn device_type(&self) -> u32 {
        self.common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.common.queue_sizes
    }

    fn features(&self) -> u64 {
        self.common.avail_features
    }

    fn ack_features(&mut self, value: u64) {
        self.common.ack_features(value);
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        match offset {
            0 if data.len() == 8 => LittleEndian::write_u64(data, self.cid),
            0 if data.len() == 4 => LittleEndian::write_u32(data, (self.cid & 0xffff_ffff) as u32),
            4 if data.len() == 4 => {
                LittleEndian::write_u32(data, ((self.cid >> 32) & 0xffff_ffff) as u32);
            }
            _ => warn!(
                "vsock: virtio-vsock received invalid read request of {} bytes at offset {}",
                data.len(),
                offset
            ),
        }
    }

    fn activate(&mut self, context: ActivationContext) -> ActivateResult {
        self.activate_inner(context)
    }

    fn reset(&mut self) {
        // Reset must converge to a fresh, inactive device even if the kernel
        // refuses to stop; log and carry on.
        if let Err(e) = self.stop_backend() {
            error!("Failed to stop vhost-kernel vsock {}: {e:?}", self.id);
        }
        self.common.reset();
        event!("virtio-device", "reset", "id", &self.id);
    }

    fn add_memory_region(
        &mut self,
        _region: &Arc<GuestRegionMmap>,
    ) -> result::Result<(), crate::Error> {
        // The memory table is rebuilt from the (already updated) guest memory
        // on activation; refresh it now if the backend is live.
        if self.running {
            self.set_mem_table()
                .map_err(crate::Error::VhostKernelVsockUpdateMemory)?;
        }
        Ok(())
    }

    fn set_access_platform(&mut self, access_platform: Arc<dyn AccessPlatform>) {
        // Rejected at creation time; nothing can enable it afterwards.
        self.common.set_access_platform(access_platform);
    }

    fn access_platform(&self) -> Option<Arc<dyn AccessPlatform>> {
        self.common.access_platform()
    }
}

impl Pausable for VhostKernelVsock {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        if self.running {
            self.vhost.stop().map_err(|e| {
                MigratableError::Pause(anyhow!("Failed to stop vhost-vsock: {e:?}"))
            })?;
        }
        self.common.pause()
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        if self.running {
            self.vhost.start().map_err(|e| {
                MigratableError::Resume(anyhow!("Failed to restart vhost-vsock: {e:?}"))
            })?;
        }
        self.common.resume()
    }
}

impl Snapshottable for VhostKernelVsock {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> result::Result<Snapshot, MigratableError> {
        // Connection state lives in the host kernel and cannot be captured.
        Err(MigratableError::Snapshot(anyhow!(
            "vsock device {} uses the vhost-kernel backend, which does not support snapshot",
            self.id
        )))
    }
}

impl Transportable for VhostKernelVsock {}

impl Migratable for VhostKernelVsock {
    fn start_dirty_log(&mut self) -> result::Result<(), MigratableError> {
        Err(MigratableError::StartDirtyLog(anyhow!(
            "vsock device {} uses the vhost-kernel backend, which does not support live migration",
            self.id
        )))
    }

    fn start_migration(&mut self) -> result::Result<(), MigratableError> {
        Err(MigratableError::StartMigration(anyhow!(
            "vsock device {} uses the vhost-kernel backend, which does not support live migration",
            self.id
        )))
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn supported_features_include_seqpacket_and_exclude_vmm_only_features() {
        assert_ne!(SUPPORTED_FEATURES & (1u64 << VIRTIO_VSOCK_F_SEQPACKET), 0);
        assert_ne!(SUPPORTED_FEATURES & (1u64 << VIRTIO_F_VERSION_1), 0);
        assert_eq!(
            SUPPORTED_FEATURES & (1u64 << crate::VIRTIO_F_ACCESS_PLATFORM),
            0
        );
        assert_eq!(SUPPORTED_FEATURES & (1u64 << crate::VIRTIO_F_IN_ORDER), 0);
    }

    #[test]
    fn creation_refuses_restore_and_access_platform_before_opening_the_device() {
        let mem = GuestMemoryAtomic::new(
            GuestMemoryMmap::from_ranges(&[(vm_memory::GuestAddress(0), 0x10_0000)]).unwrap(),
        );
        let exit_evt = EventFd::new(EFD_NONBLOCK).unwrap();
        assert!(matches!(
            VhostKernelVsock::new(
                "vsock0".into(),
                3,
                mem.clone(),
                false,
                SeccompAction::Allow,
                exit_evt.try_clone().unwrap(),
                true,
            ),
            Err(Error::RestoreUnsupported)
        ));
        assert!(matches!(
            VhostKernelVsock::new(
                "vsock0".into(),
                3,
                mem,
                true,
                SeccompAction::Allow,
                exit_evt,
                false,
            ),
            Err(Error::AccessPlatformUnsupported)
        ));
    }
}
