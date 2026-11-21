"""Run the upstream XTier console as an independent service attached to a systemd model unit.

`python -m xtier.console` starts and owns the model itself: its page follows the
state file it wrote when it launched the server. A model that is started by
systemd (here `DeepSeek-V4.1-Flash.service`, enabled at boot) is not in that
state, so the upstream page would always say OFFLINE.

This module leaves `xtier/console/` untouched. In this process only, it patches
the console's `Console` class so that, for a model whose
`configs/<name>/console.py` declares `service` (and `log`), it:

  * reads the state from `systemctl is-active` / `show`, not its state file;
  * follows the service's log for loading phases and decode telemetry;
  * starts and stops the service with `systemctl start/stop`.

The page itself is not changed: `xtier.console.ui` renders the upstream HTML, options and texts as-is.
The console has its own lifecycle: it starts before / survives the model service,
so the page can show OFFLINE / FAULT and start the model again. The systemd unit
stays the single source of truth for the launch parameters.

Run with the console's own arguments, from the repository root:

    python3 tools/console_service.py --config ds41-l20-tp2 --port 8100
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # /opt/Xtier, for `xtier` when run as a script

from xtier.console import core
from xtier.console import __main__ as cli

def _systemctl(*args, timeout=30):
    """A system `systemctl` call (the console runs as a system service, as root)."""
    return subprocess.run(['systemctl', *args], capture_output=True, text=True, timeout=timeout)


def _active(unit):
    """Whether the unit is running or on its way up."""
    try:
        return _systemctl('is-active', unit, timeout=5).stdout.strip() in ('active', 'activating', 'reloading')
    except (OSError, subprocess.TimeoutExpired):
        return False


def _started_at(unit):
    """The unit's ExecMainStartTimestamp as a Unix time (0 when unknown)."""
    try:
        text = _systemctl('show', '-p', 'ExecMainStartTimestamp', '--value', '--timestamp=unix',
                          unit, timeout=5).stdout.strip()
        return float(text[1:]) if text.startswith('@') else 0.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


class AttachedConsole(core.Console):
    """The console for one model whose lifecycle belongs to a systemd unit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attached = {k: m for k, m in self.models.items() if m.get('service')}
        if len(self.attached) > 1:
            raise SystemExit('only one model per console can declare service= (attached mode)')
        self.attach_model = next(iter(self.attached), None)

    # --- state, from systemd instead of xtier-state.json ---

    def _attach_state(self):
        model = self.attach_model
        if not model:
            return {}
        m = self.models[model]
        unit = m['service']
        if not _active(unit):
            return {}
        values = self.load_settings().get(model) or self.defaults(model)
        return {'model': model, 'options': values, 'label': self.describe(model, values),
                'runner': 'systemd-system', 'port': m['port'], 'served': m['served'],
                'log': m.get('log'), 'started_at': _started_at(unit) or time.time(),
                'ready_at': None, 'stopped_at': None}

    def read_state(self):
        return self._attach_state() if self.attach_model else super().read_state()

    def active(self, state=None):
        if self.attach_model:
            return _active(self.models[self.attach_model]['service'])
        return super().active(state)

    # --- start / stop, through systemctl ---

    def start(self, model, values):
        if model != self.attach_model:
            return super().start(model, values)
        unit = self.models[model]['service']
        result = _systemctl('start', unit, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f'systemctl start {unit} 失败：{result.stderr.strip() or result.stdout.strip()}')
        state = self._attach_state()
        if not state:   # the unit answered "already active" but is not visible yet: wait a moment
            for _ in range(20):
                time.sleep(0.2)
                state = self._attach_state()
                if state:
                    break
        return state or {'model': model, 'options': values, 'label': self.describe(model, values),
                         'runner': 'systemd-system', 'port': self.models[model]['port'],
                         'served': self.models[model]['served'], 'log': self.models[model].get('log'),
                         'started_at': time.time(), 'ready_at': None}

    def stop(self, reason=''):
        if not self.attach_model:
            return super().stop(reason)
        unit = self.models[self.attach_model]['service']
        try:
            _systemctl('stop', unit, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            pass
        for _ in range(120):
            if not _active(unit):
                return True
            time.sleep(1)
        return False

    def wait_gpu_free(self, *args, **kwargs):
        # the unit's server is the only owner of GPU0+GPU2; another service (the Qwen card) may hold GPU3
        return True if self.attach_model else super().wait_gpu_free(*args, **kwargs)

    def wait_ready(self, state, timeout):
        if not self.attach_model:
            return super().wait_ready(state, timeout)
        # no console warm-up here: the service has its own start-up, and the page only reports it
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.active(state):
                return False, '服务提前退出：' + (core.why_down(state) or ' | '.join(core.log_tail(state.get('log'), 6)))
            if core.health(state['port']):
                return True, ''
            time.sleep(3)
        return False, f'启动超时（{timeout // 60} 分钟）'

class AttachedJobs(cli.Jobs):
    """A service restart must not leave the page thinking its submitted job is still running."""

    def __init__(self, console):
        super().__init__(console)
        self.file = Path(console.state_dir) / 'xtier-console-job.json'
        try:
            saved = json.loads(self.file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            saved = {}
        if saved.get('id') and saved.get('state') == 'running':
            saved.update(state='done', message='服务已重启，当前状态见卡片', error='')
        if saved.get('id'):
            self.job = {k: saved.get(k, v) for k, v in self.job.items()}
            self.job['cancelled'] = False

    def _store(self):
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(self.job, ensure_ascii=False), encoding='utf-8')
        except OSError:
            pass

    def launch(self, model, values):
        result = super().launch(model, values)
        self._store()
        return result

    def stop(self):
        with self.lock:
            if self.job['state'] == 'running':
                self.job['cancelled'] = True
        self._store()
        return self.console.stop('手动停止')

    def _run(self, job, model, values):
        c, m = self.console, self.console.models[model]
        try:
            c.stop()
            if not c.wait_gpu_free():
                raise RuntimeError(f"GPU 上还有 {core.gpus().get('mem_used')} GiB 被其他进程占用，先停掉再启动")
            if job['cancelled']:
                raise RuntimeError('已手动停止')
            state = c.start(model, values)
            c.save_settings(model, values)
            ok, detail = c.wait_ready(state, m['ready_timeout'])
            if job['cancelled']:
                raise RuntimeError('已手动停止')
            if not ok:
                c.stop(detail)
                raise RuntimeError(detail)
            job.update(state='done',
                       message=f"{job['label']} 已就绪：http://127.0.0.1:{m['port']}/v1，模型名 {m['served']}")
        except Exception as exc:   # the page shows it
            job.update(state='failed', error=str(exc))
        finally:
            self._store()


def main(argv=None):
    # patch only in this process; xtier/console/*.py stays exactly upstream
    core.Console = AttachedConsole
    cli.Jobs = AttachedJobs
    cli.main(argv)


if __name__ == '__main__':
    main(sys.argv[1:])
