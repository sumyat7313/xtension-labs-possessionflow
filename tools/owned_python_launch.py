"""Authenticate one owned Python child, including CPython's Windows venv stub.

The caller must retain its parent lock, bind a nonce/request, enforce single-use
claims and reap its Popen handle on *any* publication/wait failure. This module
does not launch processes, grant experiment authority or search arbitrary trees.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

IS_WINDOWS = os.name == 'nt'
MAX_RECEIPT_WAIT = 10.


def windows_creationflags():
    """CREATE_NO_WINDOW | CREATE_SUSPENDED; containment must precede execution."""
    return 0x08000000 | 0x4 if IS_WINDOWS else 0


class OwnedJob:
    """Own a Windows kill-on-close job handle. Closing also kills descendants."""
    def __init__(self, kernel, handle):
        self._kernel, self._handle = kernel, handle

    def close(self):
        if self._handle is not None:
            if not self._kernel.CloseHandle(self._handle):
                raise OSError('Could not close owned containment job')
            self._handle = None


def contain_and_resume(process):
    """Contain the caller's suspended Popen child before resuming its sole thread.

    On Windows the caller MUST use windows_creationflags(). Failure never falls
    back to uncontained execution: close the job and kill/reap the owned stub.
    Keep the returned job alive and close it in finally, before waiting on failed
    work. POSIX returns None and leaves the direct-child cleanup to the caller.
    """
    if not IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),
                    ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
                    ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                    ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
    class IOCounters(ctypes.Structure):
        _fields_ = [(name,ctypes.c_uint64) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                                     'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
    class Limits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation',BasicLimits),('IoInfo',IOCounters),
                    ('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),
                    ('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
    class ThreadEntry(ctypes.Structure):
        _fields_ = [('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ThreadID',wintypes.DWORD),
                    ('th32OwnerProcessID',wintypes.DWORD),('tpBasePri',wintypes.LONG),
                    ('tpDeltaPri',wintypes.LONG),('dwFlags',wintypes.DWORD)]
    signatures = {
        'CreateJobObjectW':([ctypes.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),
        'SetInformationJobObject':([wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD],wintypes.BOOL),
        'AssignProcessToJobObject':([wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),
        'IsProcessInJob':([wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)],wintypes.BOOL),
        'CloseHandle':([wintypes.HANDLE],wintypes.BOOL),
        'CreateToolhelp32Snapshot':([wintypes.DWORD,wintypes.DWORD],wintypes.HANDLE),
        'Thread32First':([wintypes.HANDLE,ctypes.POINTER(ThreadEntry)],wintypes.BOOL),
        'Thread32Next':([wintypes.HANDLE,ctypes.POINTER(ThreadEntry)],wintypes.BOOL),
        'OpenThread':([wintypes.DWORD,wintypes.BOOL,wintypes.DWORD],wintypes.HANDLE),
        'GetProcessIdOfThread':([wintypes.HANDLE],wintypes.DWORD),
        'ResumeThread':([wintypes.HANDLE],wintypes.DWORD),
    }
    for name,(args,result) in signatures.items():
        function=getattr(kernel,name);function.argtypes=args;function.restype=result
    job=None
    try:
        _need(process.poll() is None,'Suspended owned process exited before containment')
        # This Popen handle, never a newly opened PID handle, owns the assigned process.
        _windows_receipt_parent(os.getpid(),process.pid,runtime_identity())
        handle=kernel.CreateJobObjectW(None,None)
        if not handle: raise OSError(ctypes.get_last_error(),'Cannot create containment job')
        job=OwnedJob(kernel,handle)
        limits=Limits();limits.BasicLimitInformation.LimitFlags=0x2000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            raise OSError(ctypes.get_last_error(),'Cannot set containment limits')
        if not kernel.AssignProcessToJobObject(handle,int(process._handle)):
            raise OSError(ctypes.get_last_error(),'Cannot assign owned process; uncontained execution forbidden')
        assigned=wintypes.BOOL()
        _need(kernel.IsProcessInJob(int(process._handle),handle,ctypes.byref(assigned)) and assigned.value,
              'Owned containment assignment could not be verified')
        snapshot=kernel.CreateToolhelp32Snapshot(0x4,0)
        if snapshot==ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(),'Cannot inspect suspended owned thread')
        threads=[]
        try:
            entry=ThreadEntry();entry.dwSize=ctypes.sizeof(entry)
            good=kernel.Thread32First(snapshot,ctypes.byref(entry))
            while good:
                if entry.th32OwnerProcessID==process.pid: threads.append(int(entry.th32ThreadID))
                good=kernel.Thread32Next(snapshot,ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
        _need(len(threads)==1,'Owned process must have exactly one suspended initial thread')
        thread=kernel.OpenThread(0x2|0x0800,False,threads[0])  # suspend/resume + limited query
        if not thread: raise OSError(ctypes.get_last_error(),'Cannot open owned initial thread')
        try:
            _need(kernel.GetProcessIdOfThread(thread)==process.pid,'Initial thread no longer belongs to owned process')
            _need(process.poll() is None,'Owned process exited before resume')
            _need(kernel.ResumeThread(thread)==1,'Owned initial thread was not suspended exactly once')
        finally:
            kernel.CloseHandle(thread)
        return job
    except BaseException:
        try:
            if job is not None: job.close()
        finally:
            if process.poll() is None: process.kill()
            process.wait(timeout=15)
        raise


def _need(value, message):
    if not value:
        raise ValueError(message)


def _pid(value):
    _need(type(value) is int and value > 0, 'A positive native process ID is required')
    return value


def _sha(value):
    _need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'A SHA-256 request identity is required')
    return value


def _normal(path):
    return os.path.normcase(str(Path(path).resolve()))


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def runtime_identity():
    """Identity of this interpreter environment, including its launcher bytes."""
    executable = _normal(sys.executable)
    base = _normal(getattr(sys, '_base_executable', sys.executable))
    return dict(executable=executable, base_executable=base,
                executable_sha256=_file_hash(executable), base_executable_sha256=_file_hash(base),
                prefix=_normal(sys.prefix), base_prefix=_normal(sys.base_prefix), python_version=sys.version)


def _receipt_path(path):
    path = Path(os.path.abspath(path))
    _need(path.resolve() == path, 'Linked launch receipt path is forbidden')
    for item in (path, *path.parents):
        _need(not item.is_symlink() and not (hasattr(item, 'is_junction') and item.is_junction()),
              'Linked launch receipt path is forbidden')
    return path


def _windows_process(pid):
    """Read parent PID, image and creation token from live Windows OS handles."""
    import ctypes
    from ctypes import wintypes
    pid = _pid(pid)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),
                    ('th32ProcessID',wintypes.DWORD),('th32DefaultHeapID',ctypes.c_size_t),
                    ('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),
                    ('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),
                    ('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]
    kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32FirstW.restype=wintypes.BOOL
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype=wintypes.BOOL
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    snapshot=kernel.CreateToolhelp32Snapshot(0x2,0)
    if snapshot == ctypes.c_void_p(-1).value: raise OSError(ctypes.get_last_error(),'Process snapshot unavailable')
    parent=None
    try:
        entry=Entry();entry.dwSize=ctypes.sizeof(entry)
        good=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while good:
            if entry.th32ProcessID==pid:
                parent=int(entry.th32ParentProcessID);break
            good=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    _need(parent is not None, 'Expected Windows process no longer exists')
    handle=kernel.OpenProcess(0x1000|0x100000,False,pid)  # limited query + synchronize
    if not handle: raise OSError(ctypes.get_last_error(),'Expected process cannot be authenticated')
    try:
        _need(kernel.WaitForSingleObject(handle,0)==258, 'Expected process is no longer live')
        buffer=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle,0,buffer,ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(),'Process image unavailable')
        created,exited,kernel_time,user_time=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in (created,exited,kernel_time,user_time)]):
            raise OSError(ctypes.get_last_error(),'Process creation token unavailable')
        creation=(int(created.dwHighDateTime)<<32)|int(created.dwLowDateTime)
        _need(kernel.WaitForSingleObject(handle,0)==258, 'Expected process exited during authentication')
        return dict(pid=pid,parent_pid=parent,image=_normal(buffer.value),creation_token=creation)
    finally:
        kernel.CloseHandle(handle)


def _windows_receipt_parent(parent_pid, owned_pid, runtime):
    parent=_windows_process(parent_pid);owned=_windows_process(owned_pid)
    _need(parent['image']==runtime['base_executable'], 'Parent process is not the expected Python runtime')
    _need(owned['parent_pid']==parent_pid and owned['image'] in (runtime['executable'],runtime['base_executable']),
          'Owned process is not a direct Python child')
    _need(parent['creation_token']<=owned['creation_token'], 'Owned process predates its parent')
    return parent,owned


def publish_receipt(path, process, *, request_sha256):
    """Publish ownership immediately after Popen, within the caller's cleanup try.

    The Popen object must still be live; the receipt is exclusive and never reset.
    Windows validates the actual OS child rather than trusting a claimed PID.
    """
    _sha(request_sha256);path=_receipt_path(path)
    _need(process.poll() is None,'Owned process exited before ownership publication')
    owned_pid=_pid(process.pid);parent_pid=_pid(os.getpid());runtime=runtime_identity()
    parent,owned=(None,None)
    if IS_WINDOWS:
        parent,owned=_windows_receipt_parent(parent_pid,owned_pid,runtime)
    _need(process.poll() is None,'Owned process exited during ownership publication')
    value=dict(schema_version=1,platform='windows' if IS_WINDOWS else 'posix',request_sha256=request_sha256,
               receipt_path=_normal(path),
               parent_pid=parent_pid,owned_pid=owned_pid,runtime=runtime,
               parent_identity=parent,owned_identity=owned)
    with path.open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    return value


def _wait_receipt(path, seconds):
    _need(type(seconds) in (int,float) and 0<seconds<=MAX_RECEIPT_WAIT,'Receipt wait must be bounded to ten seconds')
    deadline=time.monotonic()+seconds
    while True:
        try:
            data=_receipt_path(path).read_bytes()
        except FileNotFoundError:
            data=b''
        if data.endswith(b'\n'):
            return json.loads(data)
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError('Owned-launch receipt was not published within the bound')
        time.sleep(min(.02,remaining))


def verify_worker(path, *, request_sha256, parent_pid, expected_runtime, timeout_seconds=5.):
    """Admit this worker only as the owned PID or one exact Windows venv hop.

    POSIX requires the direct owned PID; it never admits wrappers. Nonce, parent
    lock and exclusive claim checks remain the operator's separate obligations.
    """
    _sha(request_sha256);_pid(parent_pid)
    runtime=runtime_identity()
    _need(runtime==expected_runtime,'Worker interpreter environment differs from the frozen runtime')
    value=_wait_receipt(path,timeout_seconds)
    keys={'schema_version','platform','request_sha256','receipt_path','parent_pid','owned_pid','runtime','parent_identity','owned_identity'}
    _need(isinstance(value,dict) and set(value)==keys and type(value['schema_version']) is int and value['schema_version']==1,
          'Owned-launch receipt schema differs')
    _need(value['receipt_path']==_normal(_receipt_path(path)), 'Launch receipt was moved or replayed at another path')
    _need(value['platform']==('windows' if IS_WINDOWS else 'posix') and value['request_sha256']==request_sha256 and
          type(value['parent_pid']) is int and value['parent_pid']==parent_pid and value['runtime']==runtime,
          'Owned-launch receipt is bound to a different request, parent or runtime')
    owned_pid=_pid(value['owned_pid']);worker_pid=_pid(os.getpid());worker_parent=_pid(os.getppid())
    if IS_WINDOWS:
        parent,owned=_windows_receipt_parent(parent_pid,owned_pid,runtime)
        _need(parent==value['parent_identity'] and owned==value['owned_identity'],
              'Parent/launcher creation identity changed or process was replaced')
        worker=_windows_process(worker_pid)
        _need(worker['image']==runtime['base_executable'] and worker['parent_pid']==worker_parent,
              'Worker process image or parent differs from the Python runtime')
        if worker_pid==owned_pid:
            _need(worker_parent==parent_pid and worker==owned,'Direct worker is not the owned child')
            mode='direct'
        else:
            _need(worker_parent==owned_pid and owned['image']==runtime['executable'] and
                  runtime['executable']!=runtime['base_executable'] and
                  worker['creation_token']>=owned['creation_token'],
                  'Only the exact owned Windows venv launcher hop is permitted')
            mode='windows-venv'
        # Both ancestors must remain alive with the same creation/image identity.
        _need(_windows_process(parent_pid)==parent and _windows_process(owned_pid)==owned,
              'Owned Python ancestry changed during authentication')
    else:
        _need(value['parent_identity'] is None and value['owned_identity'] is None and
              worker_pid==owned_pid and worker_parent==parent_pid,'POSIX worker must be the direct owned child')
        os.kill(parent_pid,0)  # liveness only; sends no signal
        mode='direct'
    _need(runtime_identity()==runtime,'Interpreter bytes or environment changed during authentication')
    return dict(request_sha256=request_sha256,parent_pid=parent_pid,owned_pid=owned_pid,
                worker_pid=worker_pid,mode=mode,runtime_verified=True)
