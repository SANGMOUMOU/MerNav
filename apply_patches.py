#!/usr/bin/env python3
"""
apply_patches.py — 自动应用所有补丁
====================================
Usage:
  cd /data/vepfs/users/intern/ruijie.sang/Code/WMNavigation
  # 先把 Memory_module.py 复制到 src/
  cp Memory_module.py src/Memory_module.py
  # 然后运行补丁脚本
  python apply_patches.py

会自动创建 .bak 备份。
"""
import shutil
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

def patch(filepath, pairs):
    if not os.path.exists(filepath):
        print(f'  ✗ FILE NOT FOUND: {filepath}')
        return False
    shutil.copy2(filepath, filepath + '.bak')
    with open(filepath) as f:
        content = f.read()
    ok = True
    for old, new, desc in pairs:
        if old in content:
            content = content.replace(old, new, 1)
            print(f'  ✓ {desc}')
        else:
            print(f'  ✗ NOT FOUND: {desc}')
            ok = False
    with open(filepath, 'w') as f:
        f.write(content)
    return ok

def main():
    print('=' * 60)
    print(' Applying patches: stuck fix + disk memory persistence')
    print('=' * 60)

    # ---- WMNav_agent.py ----
    agent_file = os.path.join(SRC, 'WMNav_agent.py')
    print(f'\nPatching {agent_file}...')

    # Check if 'import os' exists, add if not
    with open(agent_file) as f:
        agent_content = f.read()
    if '\nimport os\n' not in agent_content and '\nimport os ' not in agent_content:
        # Add 'import os' after 'import ast'
        if 'import ast' in agent_content:
            agent_content = agent_content.replace('import ast', 'import ast\nimport os', 1)
            with open(agent_file, 'w') as f:
                f.write(agent_content)
            print('  ✓ Added import os')

    patch(agent_file, [
        # Fix 1: Memory init with save_dir
        (
            """        # Initialize memory system
        memory_cfg = cfg.get('memory_cfg', {})
        self.memory = MemoryManager(
            vlm=None,  # Will be set after VLMs are initialized
            map_size=self.map_size,
            scale=self.scale,
            window_size=memory_cfg.get('window_size', 8),
            stuck_threshold=memory_cfg.get('stuck_threshold', 0.5)
        )""",
            """        # Initialize memory system with disk persistence
        memory_cfg = cfg.get('memory_cfg', {})
        _mem_save_dir = memory_cfg.get('save_dir', None)
        if _mem_save_dir is None:
            _mem_save_dir = os.path.join(os.environ.get("LOG_DIR", "/tmp"), 'general_memory')
        self.memory = MemoryManager(
            vlm=None,
            map_size=self.map_size,
            scale=self.scale,
            window_size=memory_cfg.get('window_size', 5),
            stuck_threshold=memory_cfg.get('stuck_threshold', 0.3),
            save_dir=_mem_save_dir
        )""",
            '[Agent 1/3] Memory init: add save_dir + lower thresholds'
        ),

        # Fix 2: Escalating stuck escape
        (
            """                if self.memory_signals.get('is_stuck', False) and self.memory.short_term.stuck_count >= 3:
                    logging.info('[Memory] Forcing turn-around due to persistent stuck state')
                    step_metadata['action_number'] = 0
                    logging_data = {'ACTION_NUMBER': 0, 'MEMORY_ESCAPE': True}""",
            """                if self.memory_signals.get('is_stuck', False) and self.memory.short_term.stuck_count >= 2:
                    _sc = self.memory.short_term.stuck_count
                    _tc = self.memory.short_term.total_stuck_count
                    logging.info(f'[Memory] Escape triggered (consecutive={_sc}, total={_tc})')
                    if _sc <= 2:
                        step_metadata['action_number'] = 0
                    elif _sc <= 4:
                        _alist = list(a_final)
                        if _alist and isinstance(_alist[0], tuple):
                            _best_idx = max(range(len(_alist)), key=lambda i: _alist[i][0])
                            step_metadata['action_number'] = _best_idx + 1
                        else:
                            step_metadata['action_number'] = 0
                    else:
                        import random as _rnd
                        _n = len(list(a_final))
                        step_metadata['action_number'] = _rnd.randint(1, max(1, _n))
                    logging_data = {'ACTION_NUMBER': step_metadata['action_number'], 'MEMORY_ESCAPE': True, 'ESCAPE_LEVEL': _sc}""",
            '[Agent 2/3] Stuck escape: 3-level escalation'
        ),

        # Fix 3: Flush memory on reset
        (
            """        self.memory.reset()
        self.memory_signals = {}""",
            """        self.memory.on_episode_end()
        self.memory.reset()
        self.memory_signals = {}""",
            '[Agent 3/3] Reset: flush memory to disk'
        ),
    ])

    # ---- WMNav_env.py ----
    env_file = os.path.join(SRC, 'WMNav_env.py')
    print(f'\nPatching {env_file}...')

    patch(env_file, [
        # Fix 1: Add direction tracking in _initialize_episode
        (
            """        self.previous_subtask = '{}'

        # Initialize memory for this episode""",
            """        self.previous_subtask = '{}'

        # Direction tracking for navigability feedback (prevents repeatedly choosing blocked directions)
        self._direction_fail_count = {}
        self._last_step_position = None
        self._last_chosen_direction = None

        # Initialize memory for this episode""",
            '[Env 1/4] Add direction tracking variables'
        ),

        # Fix 2: Add navigability feedback after goal_rotate selection
        (
            """        goal_rotate, goal_reason = self.agent.update_curiosity_value(explorable_value, reason)

        # ---- Memory-driven direction override ----""",
            """        goal_rotate, goal_reason = self.agent.update_curiosity_value(explorable_value, reason)

        # ---- NAVIGABILITY FEEDBACK: avoid choosing blocked directions ----
        if self._last_step_position is not None and self._last_chosen_direction is not None:
            _disp = np.linalg.norm(obs['agent_state'].position - self._last_step_position)
            _prev = self._last_chosen_direction
            if _disp < 0.15:
                self._direction_fail_count[_prev] = self._direction_fail_count.get(_prev, 0) + 1
                if self._direction_fail_count[_prev] >= 2:
                    logging.info(f'[NavFeedback] Dir {_prev*30}° failed {self._direction_fail_count[_prev]}x')
            else:
                self._direction_fail_count[_prev] = 0

        if hasattr(self.agent, 'memory'):
            self.agent.memory.update_direction_feedback(obs['agent_state'].position, goal_rotate)

        if explorable_value is not None and hasattr(self.agent, 'effective_mask'):
            _nav_px = {}
            for _i in range(12):
                if _i % 2 == 0:
                    continue
                _ak = str(int(_i * 30))
                _nav_px[_i] = int(np.sum(self.agent.effective_mask[_ak])) if _ak in self.agent.effective_mask else 0
            _chosen_nav = _nav_px.get(goal_rotate, 0)
            _max_nav = max(_nav_px.values()) if _nav_px else 0
            if _max_nav > 0 and _chosen_nav < _max_nav * 0.1:
                _best, _best_sc = goal_rotate, -float('inf')
                for _idx, _nav in _nav_px.items():
                    if _nav < _max_nav * 0.1:
                        continue
                    _ak = str(int(_idx * 30))
                    _cur = explorable_value.get(_ak, 0) if explorable_value else 0
                    _pen = self._direction_fail_count.get(_idx, 0) * 3.0
                    _sc = _cur * (_nav / _max_nav) - _pen
                    if _sc > _best_sc:
                        _best_sc = _sc
                        _best = _idx
                if _best != goal_rotate:
                    logging.info(f'[NavFeedback] Dir {goal_rotate*30}° blocked (nav={_chosen_nav}), -> {_best*30}° (nav={_nav_px.get(_best,0)})')
                    goal_rotate = _best
                    _ak = str(int(_best * 30))
                    if reason and _ak in reason:
                        goal_reason = reason[_ak]

            if self._direction_fail_count.get(goal_rotate, 0) >= 3:
                _cands = [(i, _nav_px.get(i, 0)) for i in range(12) if i % 2 != 0 and _nav_px.get(i, 0) > 0 and self._direction_fail_count.get(i, 0) < 2]
                if _cands:
                    _fi = max(_cands, key=lambda x: x[1])[0]
                    logging.info(f'[NavFeedback] Dir {goal_rotate*30}° failed {self._direction_fail_count[goal_rotate]}x, forced -> {_fi*30}°')
                    goal_rotate = _fi

        self._last_step_position = obs['agent_state'].position.copy()
        self._last_chosen_direction = goal_rotate

        # ---- Memory-driven direction override ----""",
            '[Env 2/4] Navigability feedback: avoid blocked directions'
        ),

        # Fix 3: Lower stuck escape threshold
        (
            """            if stuck_count >= 3:""",
            """            if stuck_count >= 2:""",
            '[Env 3/4] Stuck threshold: 3→2'
        ),

        # Fix 4: Flush memory in _post_episode
        (
            """        self.simWrapper.reset()
        self.agent.reset()""",
            """        self.simWrapper.reset()
        if hasattr(self.agent, 'memory'):
            self.agent.memory.on_episode_end()
        self.agent.reset()""",
            '[Env 4/4] Post-episode: flush memory to disk'
        ),
    ])

    print('\n' + '=' * 60)
    print(' Done! Backups saved as *.bak in src/')
    print(' Make sure Memory_module.py is copied to src/')
    print('=' * 60)


if __name__ == '__main__':
    main()