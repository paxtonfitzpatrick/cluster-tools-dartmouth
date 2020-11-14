import functools
from threading import Thread
from typing import Callable, BinaryIO, Dict, Optional, TextIO, Tuple, Union

import spur
import spurplus

from clustertools._extras.exceptions import SshProcessError
from clustertools._extras.multistream_wrapper import MultiStreamWrapper
from clustertools._extras.typing import OneOrMore, PathLike


class RemoteProcess:
    def __init__(
            self,
            command: OneOrMore[str],
            ssh_shell: Union[spurplus.SshShell, spur.SshShell],
            working_dir: Optional[PathLike] = None,
            env_updates: Optional[Dict[str, str]] = None,
            stdout: Optional[Union[OneOrMore[TextIO], OneOrMore[BinaryIO]]] = None,
            stderr: Optional[Union[OneOrMore[TextIO], OneOrMore[BinaryIO]]] = None,
            stream_encoding: Union[str, None] = 'utf-8',
            close_streams: bool = True,
            wait: bool = False,
            allow_error: bool = False,
            use_pty: bool = False,
            callback: Optional[Callable] = None,
            callback_args: Optional[Tuple] = None,
            callback_kwargs: Optional[Dict] = None
    ) -> None:
        # TODO: add docstring
        #  also note that command should be pre-formatted at this point
        self._command = command
        self._ssh_shell = ssh_shell
        self._working_dir = str(working_dir) if working_dir is not None else None
        self._stream_encoding = stream_encoding
        self._env_updates = env_updates
        self._close_streams = close_streams
        self._wait = wait
        self._allow_error = allow_error
        self._use_pty = use_pty
        self._callback = self._setup_user_callback(callback,
                                                   callback_args,
                                                   callback_kwargs)
        # attributes set later
        self.started = False
        self.completed = False
        self._proc: Optional[spur.ssh.SshProcess] = None
        self.pid: Optional[int] = None
        self._thread: Optional[Thread] = None
        self.return_code: Optional[int] = None
        # open streams as late as possible
        self.stdout = MultiStreamWrapper(stdout, encoding=stream_encoding)
        self.stderr = MultiStreamWrapper(stderr, encoding=stream_encoding)

    def _process_complete_callback(self):
        # returns once process is complete and sets self._proc._result
        self._proc.wait_for_result()
        if self._close_streams:
            self.stdout.close()
            self.stderr.close()

        self.return_code = self._proc._result.return_code
        self.completed = True
        self._callback()

    def _setup_user_callback(self, cb, cb_args, cb_kwargs):
        cb_args = cb_args or tuple()
        cb_kwargs = cb_kwargs or dict()
        if cb_args or cb_kwargs:
            if cb is None:
                raise ValueError("Callback arguments passed without callable")
            else:
                return functools.partial(cb, *cb_args, **cb_kwargs)
        elif cb:
            return cb
        return lambda: None

    def run(self):
        self._proc = self._ssh_shell.spawn(command=self._command,
                                           cwd=self._working_dir,
                                           update_env=self._env_updates,
                                           stdout=self.stdout,
                                           stderr=self.stderr,
                                           encoding=self._stream_encoding,
                                           allow_error=self._allow_error,
                                           use_pty=self._use_pty,
                                           store_pid=True)
        self.started = True
        self.pid = self._proc.pid
        if self._wait:
            self._process_complete_callback()
        else:
            self._thread = Thread(target=self._process_complete_callback,
                                  name='SshProcessMonitor',
                                  daemon=True)
            self._thread.start()

        return self

    def stdin_write(self, value):
        if self.completed:
            raise SshProcessError("Unable to send input to a completed process")
        elif not self.started:
            raise SshProcessError("The processes has not been started. "
                                  "Use 'RemoteProcess.run()' to start the process.")

        self._proc.stdin_write(value=value)

    def send_signal(self, signal):
        if self.completed:
            raise SshProcessError("Unable to send signal to a completed process")
        elif not self.started:
            raise SshProcessError("The processes has not been started. "
                                  "Use 'RemoteProcess.run()' to start the process.")

        self._proc.send_signal(signal=signal)

    def hangup(self):
        self.send_signal('SIGHUP')

    def interrupt(self):
        self.send_signal('SIGINT')

    def kill(self):
        self.send_signal('SIGKILL')

    def terminate(self):
        self.send_signal('SIGTERM')

