#!/usr/bin/env python3
"""
fix_module1.py
==============
针对已打过补丁的 WMNav_agent.py 进行模块一的修正：

问题修复：
  1. Qwen 返回 markdown fence (```json) 导致解析失败
  2. Qwen 生成过长列表被 max_tokens 截断
  3. PredictVLM 注入多余（VLM看图已有常识）

逻辑重构：
  - 模块一只获取"预期场景列表"，不注入 PredictVLM
  - 场景不匹配时：不直接封杀区域，而是提高未探索方向的 cvalue，
    引导 agent 离开当前房间去寻找预期场景

用法：
  python fix_module1.py --agent_file WMNav_agent.py

会自动备份原文件为 .bak_m1
"""

import argparse
import os
import shutil
import sys


def backup(filepath):
    bak = filepath + ".bak_m1"
    if not os.path.exists(bak):
        shutil.copy2(filepath, bak)
        print(f"  [备份] {filepath} -> {bak}")


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def write_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [写入] {filepath}")


def patch_agent(src: str) -> str:
    replacements_done = 0

    # ══════════════════════════════════════════════════════════════
    # 1. 替换 acquire_semantic_prior
    #    - 简化 prompt，限制列表长度，避免截断
    #    - 移除 PredictVLM system_instruction 注入
    # ══════════════════════════════════════════════════════════════

    old_acquire = '''    def acquire_semantic_prior(self, goal: str):
        """
        任务初始化时调用 LLM 获取目标常识先验（结构化 JSON）。
        """
        prior_prompt = (
            f"You are a household layout expert. A robot needs to find a \\"{goal}\\" inside a home. "
            f"Return a JSON object with exactly two keys:\\n"
            f"  \\"likely_rooms\\": a list of room types where a {goal} is commonly found,\\n"
            f"  \\"unlikely_rooms\\": a list of room types where a {goal} is rarely found.\\n"
            f"Be comprehensive — include any reasonable room type you can think of "
            f"(e.g., guest room, pantry, mudroom, sunroom, workshop, utility room, etc.). "
            f"Do NOT limit yourself to standard room names.\\n"
            f"Return ONLY valid JSON, no markdown, no explanation. Example:\\n"
            f'{{\\"likely_rooms\\": [\\"bedroom\\", \\"guest room\\", \\"nursery\\"], '
            f'\\"unlikely_rooms\\": [\\"kitchen\\", \\"garage\\", \\"bathroom\\"]}}'
        )
        try:
            raw_response = self._vlm_text_only(self.PlanVLM, prior_prompt)
            self.target_scene_prior = raw_response.strip()
            self._parse_prior_json(raw_response)
        except Exception as e:
            logging.warning(f"[SemanticPrior] Failed to acquire prior: {e}")
            self.target_scene_prior = ""
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []

        if self.prior_likely_rooms or self.prior_unlikely_rooms:
            likely_str = ", ".join(self.prior_likely_rooms) if self.prior_likely_rooms else "unknown"
            unlikely_str = ", ".join(self.prior_unlikely_rooms) if self.prior_unlikely_rooms else "unknown"
            prior_injection = (
                f"\\n[SEMANTIC PRIOR] The target \\"{goal}\\" is commonly found in: {likely_str}. "
                f"It is rarely found in: {unlikely_str}. "
                f"Use this knowledge to bias your exploration scores: "
                f"increase scores for directions leading toward likely rooms, "
                f"decrease scores for unlikely rooms."
            )
            if hasattr(self.PredictVLM, 'system_instruction'):
                self.PredictVLM.system_instruction = (
                    (self.PredictVLM.system_instruction or "") + prior_injection
                )
            self._prior_injected = True
            logging.info(f"[SemanticPrior] Likely: {self.prior_likely_rooms}")
            logging.info(f"[SemanticPrior] Unlikely: {self.prior_unlikely_rooms}")'''

    new_acquire = '''    def acquire_semantic_prior(self, goal: str):
        """
        任务初始化时调用 LLM 获取目标的预期场景列表。
        仅用于模块二的场景匹配判断，不注入 PredictVLM（VLM 看图时已有常识）。
        prompt 要求简短回复（每列表最多5项），避免被 max_tokens 截断。
        """
        prior_prompt = (
            f"A robot must find a \\"{goal}\\" in a home. "
            f"Return JSON: "
            f'{{\\"likely_rooms\\": [up to 5 room types where {goal} is usually found], '
            f'\\"unlikely_rooms\\": [up to 5 room types where {goal} is rarely found]}}. '
            f"Keep it SHORT. No markdown, no explanation, ONLY the JSON object."
        )
        try:
            raw_response = self._vlm_text_only(self.PlanVLM, prior_prompt)
            self.target_scene_prior = raw_response.strip()
            self._parse_prior_json(raw_response)
        except Exception as e:
            logging.warning(f"[SemanticPrior] Failed to acquire prior: {e}")
            self.target_scene_prior = ""
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []

        if self.prior_likely_rooms or self.prior_unlikely_rooms:
            logging.info(f"[SemanticPrior] Likely: {self.prior_likely_rooms}")
            logging.info(f"[SemanticPrior] Unlikely: {self.prior_unlikely_rooms}")
        else:
            logging.warning(f"[SemanticPrior] No prior acquired for \\'{goal}\\'")'''

    if old_acquire in src:
        src = src.replace(old_acquire, new_acquire, 1)
        replacements_done += 1
        print("  [1/3] ✓ acquire_semantic_prior 已替换")
    else:
        print("  [1/3] ✗ 未找到 acquire_semantic_prior 原始代码，尝试宽松匹配...")
        # 尝试匹配不带转义的版本
        alt_old = 'def acquire_semantic_prior(self, goal: str):'
        if alt_old in src:
            # 找到函数起始和下一个同级 def 之间的内容
            start = src.index(alt_old)
            # 找到这个方法的缩进级别
            line_start = src.rfind('\n', 0, start) + 1
            indent = start - line_start
            # 找到下一个同缩进级别的 def
            search_from = start + len(alt_old)
            next_def = src.find('\n' + ' ' * indent + 'def ', search_from)
            if next_def == -1:
                next_def = len(src)
            old_method = src[line_start:next_def]
            # 构建新方法（保持相同缩进）
            new_method_lines = new_acquire.splitlines()
            src = src[:line_start] + new_acquire + '\n' + src[next_def:]
            replacements_done += 1
            print("  [1/3] ✓ acquire_semantic_prior 已通过宽松匹配替换")
        else:
            print("  [1/3] ✗ 完全找不到 acquire_semantic_prior，跳过")

    # ══════════════════════════════════════════════════════════════
    # 2. 替换 _parse_prior_json
    #    - 处理 markdown code fence (```json ... ```)
    #    - 处理被截断的 JSON（没有闭合括号）
    # ══════════════════════════════════════════════════════════════

    old_parse = '''    def _parse_prior_json(self, raw_response: str):
        """从 LLM 响应中提取结构化 likely_rooms / unlikely_rooms。"""
        text = raw_response.strip()
        # 去除 markdown code fence
        if text.startswith("```"):
            text = text.split("\\n", 1)[-1]  # 去掉第一行 ```json
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        brace_start = text.find('{')
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        text = text[brace_start:i+1]
                        break

        data = None
        try:
            data = _json.loads(text)
        except:
            pass
        if data is None:
            try:
                data = ast.literal_eval(text)
            except:
                pass
        if data is None:
            try:
                data = _json.loads(text.replace("'", '"'))
            except:
                logging.warning(f"[SemanticPrior] Cannot parse prior JSON: {raw_response[:200]}")
                self.prior_likely_rooms = []
                self.prior_unlikely_rooms = []
                return

        if isinstance(data, dict):
            self.prior_likely_rooms = [str(r).strip().lower() for r in data.get('likely_rooms', [])]
            self.prior_unlikely_rooms = [str(r).strip().lower() for r in data.get('unlikely_rooms', [])]
        else:
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []'''

    new_parse = '''    def _parse_prior_json(self, raw_response: str):
        """
        从 LLM 响应中提取 likely_rooms / unlikely_rooms。
        鲁棒处理：markdown fence、被截断的 JSON、单引号、多余文本。
        """
        import re as _re
        text = raw_response.strip()

        # 1) 去除 markdown code fence
        text = _re.sub(r'^```(?:json)?\\s*', '', text)
        text = _re.sub(r'\\s*```$', '', text)
        text = text.strip()

        # 2) 提取 JSON 块
        brace_start = text.find('{')
        if brace_start == -1:
            logging.warning(f"[SemanticPrior] No JSON found in: {raw_response[:150]}")
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []
            return

        # 寻找匹配的闭合大括号
        depth = 0
        brace_end = -1
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break

        if brace_end != -1:
            # 正常闭合
            text = text[brace_start:brace_end + 1]
        else:
            # JSON 被截断（没有闭合括号）→ 尝试修复
            text = text[brace_start:]
            logging.warning(f"[SemanticPrior] JSON truncated, attempting repair...")
            # 去掉最后一个不完整的元素（截断处通常在引号中间）
            # 策略：找最后一个完整的引号字符串后截断
            last_good = max(text.rfind('",'), text.rfind("',"))
            if last_good > 0:
                text = text[:last_good + 1]
            # 补齐未闭合的括号
            open_sq = text.count('[') - text.count(']')
            open_br = text.count('{') - text.count('}')
            text += ']' * max(open_sq, 0) + '}' * max(open_br, 0)

        # 3) 多轮尝试解析
        data = None
        # 尝试1: 直接 json.loads
        try:
            data = _json.loads(text)
        except:
            pass
        # 尝试2: 单引号转双引号
        if data is None:
            try:
                data = _json.loads(text.replace("'", '"'))
            except:
                pass
        # 尝试3: ast.literal_eval
        if data is None:
            try:
                data = ast.literal_eval(text)
            except:
                pass

        if data is None:
            logging.warning(f"[SemanticPrior] Cannot parse prior JSON: {raw_response[:200]}")
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []
            return

        if isinstance(data, dict):
            self.prior_likely_rooms = list(set(
                str(r).strip().lower() for r in data.get('likely_rooms', []) if str(r).strip()
            ))
            self.prior_unlikely_rooms = list(set(
                str(r).strip().lower() for r in data.get('unlikely_rooms', []) if str(r).strip()
            ))
        else:
            self.prior_likely_rooms = []
            self.prior_unlikely_rooms = []'''

    if old_parse in src:
        src = src.replace(old_parse, new_parse, 1)
        replacements_done += 1
        print("  [2/3] ✓ _parse_prior_json 已替换")
    else:
        print("  [2/3] ✗ 未找到 _parse_prior_json 原始代码")
        # 尝试定位并替换
        if 'def _parse_prior_json(self, raw_response: str):' in src:
            marker = 'def _parse_prior_json(self, raw_response: str):'
            start = src.index(marker)
            line_start = src.rfind('\n', 0, start) + 1
            indent = start - line_start
            search_from = start + len(marker)
            next_def = src.find('\n' + ' ' * indent + 'def ', search_from)
            if next_def == -1:
                next_def = len(src)
            src = src[:line_start] + new_parse + '\n' + src[next_def:]
            replacements_done += 1
            print("  [2/3] ✓ _parse_prior_json 已通过定位替换")
        else:
            print("  [2/3] ✗ 完全找不到 _parse_prior_json，跳过")

    # ══════════════════════════════════════════════════════════════
    # 3. 替换 check_scene_mismatch_and_block
    #    - 不再直接封杀区域（cvalue=0.1）
    #    - 改为：提高未探索方向的 cvalue，引导离开去找预期场景
    # ══════════════════════════════════════════════════════════════

    old_mismatch = '''    def check_scene_mismatch_and_block(self, agent_state):
        if self.steps_since_door_entry < 3:
            return False
        if not self._is_scene_mismatch():
            return False

        logging.info(f"[WorkingMemory] Scene MISMATCH -> blocking, scene='{self.current_scene_type}'")
        self._room_mismatch_blocked = True

        agent_coords = self._global_to_grid(agent_state.position)
        radius_px = int(2.0 * self.scale)
        x, y = agent_coords
        h, w = self.cvalue_map.shape[:2]
        y1, y2 = max(0, y - radius_px), min(h, y + radius_px)
        x1, x2 = max(0, x - radius_px), min(w, x + radius_px)
        self.cvalue_map[y1:y2, x1:x2] = 0.1

        if self.door_memory:
            last_door = self.door_memory[-1]
            door_grid = self._global_to_grid(np.array([last_door[0], agent_state.position[1], last_door[1]]))
            dr = int(1.0 * self.scale)
            dy1, dy2 = max(0, door_grid[1] - dr), min(h, door_grid[1] + dr)
            dx1, dx2 = max(0, door_grid[0] - dr), min(w, door_grid[0] + dr)
            self.cvalue_map[dy1:dy2, dx1:dx2] = 10.0
        return True'''

    new_mismatch = '''    def check_scene_mismatch_and_block(self, agent_state):
        """
        场景不匹配处理（重构版）：
        不直接封杀当前区域，而是：
        1. 降低当前已探索区域的 cvalue（降权但不清零，仍允许路过）
        2. 提高所有未探索方向的 cvalue（引导去找预期场景）
        3. 如果有门记忆，优先引导回到门的方向（去其他房间找预期场景）
        """
        if self.steps_since_door_entry < 3:
            return False
        if not self._is_scene_mismatch():
            return False

        logging.info(
            f"[WorkingMemory] Scene MISMATCH: current='{self.current_scene_type}', "
            f"expected={self.prior_likely_rooms}. Prioritizing unexplored areas."
        )
        self._room_mismatch_blocked = True

        # 1) 降低当前区域 cvalue（降权，不是清零）
        agent_coords = self._global_to_grid(agent_state.position)
        radius_px = int(2.0 * self.scale)
        x, y = agent_coords
        h, w = self.cvalue_map.shape[:2]
        y1, y2 = max(0, y - radius_px), min(h, y + radius_px)
        x1, x2 = max(0, x - radius_px), min(w, x + radius_px)
        # 将已探索的当前区域 cvalue 减半（最低到 1.0），而非直接置 0.1
        region = self.cvalue_map[y1:y2, x1:x2].astype(np.float32)
        region = np.maximum(region * 0.5, 1.0)
        self.cvalue_map[y1:y2, x1:x2] = region.astype(np.float16)

        # 2) 提高未探索方向的 cvalue（引导 agent 离开去寻找预期场景）
        for angle_key, mask in self.panoramic_mask.items():
            if not np.any(mask):
                continue
            # 检查该方向的 explored_map 覆盖率
            direction_explored = self.explored_map[mask]
            explored_count = np.all(direction_explored == self.explored_color, axis=-1).sum()
            total_count = max(mask.sum(), 1)
            explore_ratio = explored_count / total_count
            # 未充分探索的方向 → 提高 cvalue
            if explore_ratio < 0.4:
                current_vals = self.cvalue_map[mask].astype(np.float32)
                boosted = np.minimum(current_vals + 3.0, 10.0)
                self.cvalue_map[mask] = boosted.astype(np.float16)
                logging.info(
                    f"[WorkingMemory] Boosted unexplored direction {angle_key}deg "
                    f"(explore_ratio={explore_ratio:.2f})"
                )

        # 3) 如果有门记忆，强力引导回到最近的门方向（去其他房间）
        if self.door_memory:
            last_door = self.door_memory[-1]
            door_grid = self._global_to_grid(
                np.array([last_door[0], agent_state.position[1], last_door[1]])
            )
            dr = int(1.5 * self.scale)
            dy1, dy2 = max(0, door_grid[1] - dr), min(h, door_grid[1] + dr)
            dx1, dx2 = max(0, door_grid[0] - dr), min(w, door_grid[0] + dr)
            self.cvalue_map[dy1:dy2, dx1:dx2] = 10.0
            logging.info("[WorkingMemory] Door direction boosted to 10.0")

        return True'''

    if old_mismatch in src:
        src = src.replace(old_mismatch, new_mismatch, 1)
        replacements_done += 1
        print("  [3/3] ✓ check_scene_mismatch_and_block 已替换")
    else:
        print("  [3/3] ✗ 未找到 check_scene_mismatch_and_block 原始代码")
        if 'def check_scene_mismatch_and_block(self, agent_state):' in src:
            marker = 'def check_scene_mismatch_and_block(self, agent_state):'
            start = src.index(marker)
            line_start = src.rfind('\n', 0, start) + 1
            indent = start - line_start
            search_from = start + len(marker)
            next_def = src.find('\n' + ' ' * indent + 'def ', search_from)
            if next_def == -1:
                next_def = len(src)
            src = src[:line_start] + new_mismatch + '\n' + src[next_def:]
            replacements_done += 1
            print("  [3/3] ✓ check_scene_mismatch_and_block 已通过定位替换")
        else:
            print("  [3/3] ✗ 完全找不到 check_scene_mismatch_and_block，跳过")

    print(f"\n  共完成 {replacements_done}/3 处替换")
    return src


def main():
    parser = argparse.ArgumentParser(description="修正模块一：语义先验 & 场景匹配逻辑")
    parser.add_argument("--agent_file", default="WMNav_agent.py")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  模块一修正补丁")
    print("  [1] 简化 prompt，避免截断")
    print("  [2] 鲁棒 JSON 解析（markdown fence + 截断修复）")
    print("  [3] 场景不匹配 → 优先探索未知区域（非封杀）")
    print("=" * 60)

    if not os.path.exists(args.agent_file):
        print(f"\n  错误: 未找到 {args.agent_file}")
        sys.exit(1)

    print(f"\n  正在修改 {args.agent_file} ...")
    if not args.no_backup:
        backup(args.agent_file)

    src = read_file(args.agent_file)
    src = patch_agent(src)
    write_file(args.agent_file, src)

    print("\n" + "=" * 60)
    print("  补丁完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()