"""The first process inside the agent's sandbox (scripts/agent_run.py starts it under bwrap, in a network namespace of
its own with loopback only). It forwards 127.0.0.1:PORT to the egress proxy's unix socket, the sandbox's only way
out (the agent CLI reaches its model API through it), runs the agent command, and exits with its status. When the
runner writes to (or closes) STOP_FD, the agent gets SIGTERM, then SIGKILL after GRACE seconds.

Before the agent it starts HOST_COMMAND, the program host (program_host.py), with its stdout on HOST_LOG_FD and the
proxy variables removed from its environment, as are the variables SANDBOX_PRIVATE_VARS names (a model provider's
key, for the agent CLI alone) and SANDBOX_PRIVATE_VARS itself; it starts the agent's robot programs so that they
outlive the command that asked. Once the agent has ended, the host gets SIGTERM (it stops those programs), then
SIGKILL after HOST_GRACE seconds, so no program is left running when the runner evaluates the task.

Only processes in this process's own PID namespace are forwarded: the agent CLI, not the commands it runs, which
Codex starts in a sandbox with a PID namespace of its own (they share this network namespace, so they can reach the
port). Other connections are closed, and reported on stderr. The program host shares this PID namespace but not
the proxy password; the programs it starts run in command sandboxes like the commands.

    python3 sandbox_init.py PORT PROXY_SOCKET STOP_FD HOST_LOG_FD HOST_COMMAND... -- COMMAND...

Standard library only: it runs with the system Python.
"""
import os
import signal
import socket
import subprocess
import sys
import threading

GRACE = 15.0
HOST_GRACE = 15.0
PROXY_VARS = {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"}


def _forward(conn, proxy_path):
    """One connection: bytes both ways until each side has finished sending; the second direction to end closes."""
    try:
        up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        up.connect(proxy_path)
    except OSError:
        conn.close()
        return
    left, lock = [2], threading.Lock()

    def pump(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        with lock:
            left[0] -= 1
            last = left[0] == 0
        if last:
            conn.close()
            up.close()
    threading.Thread(target=pump, args=(conn, up), daemon=True).start()
    threading.Thread(target=pump, args=(up, conn), daemon=True).start()


def _owner(conn, port):
    """The PID of the process in our PID namespace that holds the client end of conn, or None."""
    peer = conn.getpeername()[1]
    inode = None
    with open("/proc/net/tcp") as f:
        for line in f.readlines()[1:]:
            fields = line.split()
            local, remote = fields[1].split(":"), fields[2].split(":")
            if int(local[1], 16) == peer and int(remote[1], 16) == port:
                inode = fields[9]
                break
    if inode is None:
        return None
    target, ours = f"socket:[{inode}]", os.readlink("/proc/self/ns/pid")
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.readlink(f"/proc/{pid}/ns/pid") != ours:
                continue
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:                                # an fd closed mid-scan skips that fd, not the whole process
                if os.readlink(f"/proc/{pid}/fd/{fd}") == target:
                    return int(pid)
            except OSError:
                continue
    return None


def _serve(srv, port, proxy_path):
    while True:
        conn, _ = srv.accept()
        if _owner(conn, port) is None:
            print(f"sandbox_init: refused a proxy connection from outside the agent's PID namespace "
                  f"(port {conn.getpeername()[1]})", file=sys.stderr, flush=True)
            conn.close()
            continue
        _forward(conn, proxy_path)


def main():
    sep = sys.argv.index("--")
    port, proxy_path, stop_fd, host_log = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    host_cmd, cmd = sys.argv[5:sep], sys.argv[sep + 1:]
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)          # listening before the agent starts
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    threading.Thread(target=_serve, args=(srv, port, proxy_path), daemon=True).start()
    private = PROXY_VARS | set(os.environ.get("SANDBOX_PRIVATE_VARS", "").split()) | {"SANDBOX_PRIVATE_VARS"}
    host = subprocess.Popen(host_cmd, stdin=subprocess.DEVNULL, stdout=host_log,
                            env={k: v for k, v in os.environ.items() if k not in private})
    os.close(host_log)
    child = subprocess.Popen(cmd)

    def stop():
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(GRACE)
            except subprocess.TimeoutExpired:
                child.kill()

    def watch_stop():
        try:
            os.read(stop_fd, 1)                          # data or EOF: the runner wants the agent stopped
        except OSError:
            return
        stop()
    threading.Thread(target=watch_stop, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: threading.Thread(target=stop, daemon=True).start())
    rc = child.wait()
    host.terminate()                                     # the programs it started stop now
    try:
        host.wait(HOST_GRACE)
    except subprocess.TimeoutExpired:
        print(f"sandbox_init: the program host did not exit {HOST_GRACE:.0f} s after SIGTERM: killed",
              file=sys.stderr, flush=True)
        host.kill()
        host.wait()
    sys.exit(rc if rc >= 0 else 128 - rc)


if __name__ == "__main__":
    main()
