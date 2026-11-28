// Copyright 2026 The Cloud Hypervisor Authors. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

//! Disk image factory.
//!
//! [`open_disk`] is the single entry point for opening a disk image.
//! It opens the file, validates the image type, probes async I/O
//! support, and constructs the appropriate backend. Callers receive
//! a trait object that is ready for use by virtio queue workers.

use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::path::Path;
use std::sync::OnceLock;
use std::{fs, io};

use log::{info, warn};

#[cfg(feature = "io_uring")]
use crate::block_io_uring_is_supported;
use crate::disk_file::AsyncFullDiskFile;
use crate::error::{BlockError, BlockErrorKind, BlockResult};
use crate::formats::qcow::QcowDisk;
use crate::formats::raw::{RawBackend, RawDisk};
use crate::formats::vhd::VhdDisk;
use crate::formats::vhdx::VhdxDisk;
use crate::formats::vmdk::VmdkDisk;
use crate::{
    ImageType, block_aio_is_supported, open_disk_image, preallocate_disk, validate_image_type,
};

/// Options for opening a disk image via [`open_disk`].
pub struct DiskOpenOptions<'a> {
    pub path: &'a Path,
    pub readonly: bool,
    pub direct: bool,
    pub sparse: bool,
    pub backing_files: bool,
    pub disable_io_uring: bool,
    pub disable_aio: bool,
}

/// Returns true when io_uring is supported on the running kernel.
///
/// The result is cached so the probe runs at most once per process.
#[cfg(feature = "io_uring")]
fn io_uring_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(block_io_uring_is_supported)
}

/// Returns true when Linux AIO is supported on the running kernel.
///
/// The result is cached so the probe runs at most once per process.
fn aio_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(block_aio_is_supported)
}

/// Open a disk image and construct the appropriate async backend.
///
/// - Opens the file with the requested access mode and flags.
/// - Checks specified format against detected format.
/// - Probes io_uring and Linux AIO support on the running kernel.
/// - Constructs the most capable backend available for the format,
///   preferring io_uring over AIO over synchronous fallback.
pub fn open_disk(
    options: &DiskOpenOptions<'_>,
    image_type: ImageType,
) -> BlockResult<Box<dyn AsyncFullDiskFile>> {
    let mut fs_options = fs::OpenOptions::new();
    fs_options.read(true);
    fs_options.write(!options.readonly);
    let mut flags = 0;
    if options.direct {
        flags |= libc::O_DIRECT;
    }
    // Claim a block device exclusively: the kernel arbitrates by device, so a second
    // node or path naming the same device cannot be claimed by another O_EXCL opener.
    if !options.readonly
        && fs::metadata(options.path).is_ok_and(|m| m.file_type().is_block_device())
    {
        flags |= libc::O_EXCL;
    }
    if flags != 0 {
        fs_options.custom_flags(flags);
    }

    let mut file = open_disk_image(options.path, &fs_options)?;
    if !validate_image_type(&mut file, image_type)? {
        return Err(BlockError::from_kind(BlockErrorKind::ImageTypeMismatch {
            specified: image_type,
        })
        .with_path(options.path));
    }

    let disk: Box<dyn AsyncFullDiskFile> = match image_type {
        ImageType::FixedVhd => open_fixed_vhd(file, options)?,
        ImageType::Raw => open_raw(file, options)?,
        ImageType::Qcow2 => open_qcow2(file, options)?,
        ImageType::Vhdx => open_vhdx(file, options)?,
        ImageType::FlatVmdk => open_flat_vmdk(file, options)?,
        ImageType::Unknown => unreachable!(),
    };

    Ok(disk)
}

fn open_vhdx(
    file: fs::File,
    options: &DiskOpenOptions<'_>,
) -> BlockResult<Box<dyn AsyncFullDiskFile>> {
    info!("Opening VHDX disk file with synchronous backend");
    Ok(Box::new(
        VhdxDisk::new(file, options.direct).map_err(|e| e.with_path(options.path))?,
    ))
}

fn open_fixed_vhd(
    file: fs::File,
    options: &DiskOpenOptions<'_>,
) -> BlockResult<Box<dyn AsyncFullDiskFile>> {
    #[cfg(feature = "io_uring")]
    if !options.disable_io_uring {
        if io_uring_supported() {
            info!("Opening fixed VHD disk file with io_uring backend");
            return Ok(Box::new(
                VhdDisk::new(file, true, options.direct).map_err(|e| e.with_path(options.path))?,
            ));
        }
        warn!("io_uring runtime probe failed for fixed VHD, using synchronous backend");
    }

    info!("Opening fixed VHD disk file with synchronous backend");
    Ok(Box::new(
        VhdDisk::new(file, false, options.direct).map_err(|e| e.with_path(options.path))?,
    ))
}

fn open_raw(
    file: fs::File,
    options: &DiskOpenOptions<'_>,
) -> BlockResult<Box<dyn AsyncFullDiskFile>> {
    if !options.readonly && !options.sparse {
        preallocate_disk(&file, options.path);
    }

    #[cfg(feature = "io_uring")]
    if !options.disable_io_uring {
        if io_uring_supported() {
            info!("Opening RAW disk file with io_uring backend");
            return Ok(Box::new(RawDisk::new(
                file,
                RawBackend::IoUring,
                options.direct,
            )));
        }
        warn!("io_uring runtime probe failed for RAW, trying next backend");
    }

    if !options.disable_aio {
        if aio_supported() {
            info!("Opening RAW disk file with AIO backend");
            return Ok(Box::new(RawDisk::new(
                file,
                RawBackend::Aio,
                options.direct,
            )));
        }
        warn!("AIO runtime probe failed for RAW, using synchronous backend");
    }

    info!("Opening RAW disk file with synchronous backend");
    Ok(Box::new(RawDisk::new(
        file,
        RawBackend::Sync,
        options.direct,
    )))
}

fn open_qcow2(
    file: fs::File,
    options: &DiskOpenOptions<'_>,
) -> BlockResult<Box<dyn AsyncFullDiskFile>> {
    #[cfg(feature = "io_uring")]
    if !options.disable_io_uring {
        if io_uring_supported() {
            info!("Opening QCOW2 disk file with io_uring backend");
            return Ok(Box::new(
                QcowDisk::new(
                    file,
                    options.direct,
                    options.backing_files,
                    options.sparse,
                    true,
                )
                .map_err(|e| e.with_path(options.path))?,
            ));
        }
        warn!("io_uring runtime probe failed for QCOW2, using synchronous backend");
    }

    info!("Opening QCOW2 disk file with synchronous backend");
    Ok(Box::new(
        QcowDisk::new(
            file,
            options.direct,
            options.backing_files,
            options.sparse,
            false,
        )
        .map_err(|e| e.with_path(options.path))?,
    ))
}

fn open_flat_vmdk(
    file: fs::File,
    options: &DiskOpenOptions<'_>,
) -> BlockResult<Box<dyn AsyncFullDiskFile>> {
    if !options.backing_files {
        return Err(BlockError::new(
            BlockErrorKind::UnsupportedFeature,
            io::Error::new(
                io::ErrorKind::Unsupported,
                "VMDK images require backing files enabled",
            ),
        )
        .with_path(options.path));
    }

    info!("Opening VMDK disk file with synchronous backend");
    Ok(Box::new(
        VmdkDisk::new(file, options.path, options.readonly, options.direct)
            .map_err(|e| e.with_path(options.path))?,
    ))
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::formats::qcow;

    fn default_options(path: &Path) -> DiskOpenOptions<'_> {
        DiskOpenOptions {
            path,
            readonly: false,
            direct: false,
            sparse: false,
            backing_files: false,
            disable_io_uring: true,
            disable_aio: true,
        }
    }

    #[test]
    fn nonexistent_path_returns_error() {
        let path = Path::new("/tmp/no_such_disk_image.raw");
        let options = default_options(path);
        match open_disk(&options, ImageType::Raw) {
            Err(e) => assert_eq!(e.kind(), BlockErrorKind::Io),
            Ok(_) => panic!("expected error for nonexistent path"),
        }
    }

    #[test]
    fn detect_qcow2_image() {
        let tmp = qcow::QcowTempDisk::new(100 * 1024 * 1024, None, false, true, false)
            .unwrap()
            .into_tempfile();
        let path = tmp.as_path().to_owned();
        let options = default_options(&path);
        open_disk(&options, ImageType::Qcow2).unwrap();
    }

    #[test]
    fn mismatched_image_type_returns_error() {
        let tmp = TempFile::new().unwrap();
        tmp.as_file().set_len(1 << 20).unwrap();
        let path = tmp.as_path().to_owned();
        let options = default_options(&path);
        match open_disk(&options, ImageType::Qcow2) {
            Err(e) => assert_eq!(
                e.kind(),
                BlockErrorKind::ImageTypeMismatch {
                    specified: ImageType::Qcow2,
                }
            ),
            Ok(_) => panic!("expected error for mismatched image type"),
        }
    }

    #[test]
    fn open_readonly() {
        let tmp = TempFile::new().unwrap();
        tmp.as_file().set_len(1 << 20).unwrap();
        let path = tmp.as_path().to_owned();
        let mut options = default_options(&path);
        options.readonly = true;
        open_disk(&options, ImageType::Raw).unwrap();
    }

    #[test]
    fn sync_fallback_when_async_disabled() {
        let tmp = TempFile::new().unwrap();
        let size = 1u64 << 20;
        tmp.as_file().set_len(size).unwrap();
        let path = tmp.as_path().to_owned();
        let options = DiskOpenOptions {
            path: &path,
            readonly: false,
            direct: false,
            sparse: false,
            backing_files: false,
            disable_io_uring: true,
            disable_aio: true,
        };
        let disk = open_disk(&options, ImageType::Raw).unwrap();
        assert_eq!(disk.logical_size().unwrap(), size);
    }

    #[test]
    fn explicit_raw_type_does_not_depend_on_header_bytes() {
        // A guest owns the bytes of its data disk, including the first sector.
        // An explicit `raw` must open whatever those bytes are.
        for header in [
            &0x5146_49fb_u32.to_be_bytes()[..],
            &0x656c_6966_7864_6876_u64.to_le_bytes()[..],
            b"# Disk DescriptorFile\nversion=1\n",
        ] {
            let tmp = TempFile::new().unwrap();
            tmp.as_file().set_len(1 << 20).unwrap();
            tmp.as_file().write_all_at(header, 0).unwrap();
            let path = tmp.as_path().to_owned();
            let options = default_options(&path);
            let disk = open_disk(&options, ImageType::Raw).unwrap();
            assert_eq!(disk.logical_size().unwrap(), 1 << 20);
        }
    }

    /// A loop device over a temporary file, detached on drop. Needs root.
    struct LoopDevice {
        path: std::path::PathBuf,
        _backing: TempFile,
    }

    impl LoopDevice {
        const LOOP_SET_FD: libc::c_ulong = 0x4C00;
        const LOOP_CLR_FD: libc::c_ulong = 0x4C01;
        const LOOP_CTL_GET_FREE: libc::c_ulong = 0x4C82;

        fn new() -> Self {
            let backing = TempFile::new().unwrap();
            backing.as_file().set_len(1 << 20).unwrap();
            let control = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/loop-control")
                .expect("a loop device needs root and /dev/loop-control");
            // Tests in other processes ask for a free device at the same time and
            // can be handed the same one: the loser of the attach gets EBUSY and asks again.
            for _ in 0..64 {
                // SAFETY: LOOP_CTL_GET_FREE takes no argument and returns a device number.
                let number = unsafe { libc::ioctl(control.as_raw_fd(), Self::LOOP_CTL_GET_FREE) };
                assert!(number >= 0, "{}", io::Error::last_os_error());
                let path = std::path::PathBuf::from(format!("/dev/loop{number}"));
                let device = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .unwrap();
                // SAFETY: LOOP_SET_FD takes the backing file descriptor.
                let rc = unsafe {
                    libc::ioctl(
                        device.as_raw_fd(),
                        Self::LOOP_SET_FD,
                        backing.as_file().as_raw_fd(),
                    )
                };
                if rc == 0 {
                    return Self {
                        path,
                        _backing: backing,
                    };
                }
                let error = io::Error::last_os_error();
                assert_eq!(error.raw_os_error(), Some(libc::EBUSY), "{error}");
            }
            panic!("no free loop device after 64 attempts");
        }
    }

    impl Drop for LoopDevice {
        fn drop(&mut self) {
            if let Ok(device) = fs::OpenOptions::new().read(true).open(&self.path) {
                // SAFETY: LOOP_CLR_FD takes no argument.
                unsafe { libc::ioctl(device.as_raw_fd(), Self::LOOP_CLR_FD) };
            }
        }
    }

    #[test]
    fn writable_block_device_is_opened_exclusively() {
        let device = LoopDevice::new();
        let options = default_options(&device.path);

        let holder = open_disk(&options, ImageType::Raw).unwrap();
        // A second writable open of the same device, by this path or any other
        // node naming it, is refused by the kernel while the first holds it.
        match open_disk(&options, ImageType::Raw) {
            Err(e) => {
                assert_eq!(e.kind(), BlockErrorKind::Io);
                let os_error = e
                    .downcast_ref::<io::Error>()
                    .and_then(io::Error::raw_os_error);
                assert_eq!(os_error, Some(libc::EBUSY));
            }
            Ok(_) => panic!("a held block device was opened a second time"),
        }
        drop(holder);
        open_disk(&options, ImageType::Raw).unwrap();
    }

    #[test]
    fn readonly_block_device_is_not_claimed() {
        let device = LoopDevice::new();
        let mut options = default_options(&device.path);
        let holder = open_disk(&options, ImageType::Raw).unwrap();
        options.readonly = true;
        open_disk(&options, ImageType::Raw).unwrap();
        drop(holder);
    }
}
