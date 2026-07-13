import json
import random
from pathlib import Path
from typing import Dict, List, Optional

VISUAL_GROUPS_PATH = Path('/home/xp/fed-multimodal/fed_multimodal/Local/ucf101_visual_pattern_groups.json')
ATTACK_GROUPS_V2_PATH = Path(__file__).with_name('ucf101_attack_groups_v2.json')

MODE_DISPLAY_NAMES = {
    'clean': {'zh': '干净基线', 'en': 'Clean baseline'},
    'gan_same_group_shift': {'zh': '组内生成特征攻击', 'en': 'Intra-group generated feature attack'},
    'cross_modal_mismatch_eval': {'zh': '组外跨模态攻击', 'en': 'Inter-group cross-modal attack'},
}

_FIXED_ATTACKED_LABEL_NAMES = [
    'ApplyEyeMakeup',
    'ApplyLipstick',
    'HandstandPushups',
    'HandstandWalking',
    'CricketBowling',
    'CricketShot',
    'PlayingCello',
    'PlayingSitar',
    'BoxingPunchingBag',
    'BoxingSpeedBag',
    'HammerThrow',
    'Shotput',
]

_FIXED_SAME_GROUP_TARGET_NAMES = {
    'ApplyEyeMakeup': 'ApplyLipstick',
    'ApplyLipstick': 'ApplyEyeMakeup',
    'HandstandPushups': 'HandstandWalking',
    'HandstandWalking': 'HandstandPushups',
    'CricketBowling': 'CricketShot',
    'CricketShot': 'CricketBowling',
    'PlayingCello': 'PlayingSitar',
    'PlayingSitar': 'PlayingCello',
    'BoxingPunchingBag': 'BoxingSpeedBag',
    'BoxingSpeedBag': 'BoxingPunchingBag',
    'HammerThrow': 'Shotput',
    'Shotput': 'HammerThrow',
}

_FIXED_CROSS_GROUP_TARGET_NAMES = {
    'ApplyEyeMakeup': 'HammerThrow',
    'ApplyLipstick': 'Shotput',
    'HandstandPushups': 'PlayingCello',
    'HandstandWalking': 'PlayingSitar',
    'CricketBowling': 'ApplyEyeMakeup',
    'CricketShot': 'ApplyLipstick',
    'PlayingCello': 'BoxingPunchingBag',
    'PlayingSitar': 'BoxingSpeedBag',
    'BoxingPunchingBag': 'CricketBowling',
    'BoxingSpeedBag': 'CricketShot',
    'HammerThrow': 'HandstandPushups',
    'Shotput': 'HandstandWalking',
}


def load_visual_groups() -> Dict:
    with open(VISUAL_GROUPS_PATH, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def sorted_label_names() -> List[str]:
    data = load_visual_groups()
    return sorted(data['all_labels'])


def label_name_to_id() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted_label_names())}


def label_id_to_name() -> Dict[int, str]:
    mapping = label_name_to_id()
    return {idx: name for name, idx in mapping.items()}


def label_to_group() -> Dict[str, str]:
    return load_visual_groups()['label_to_group']


def load_attack_group_spec(spec_path: Optional[Path] = None) -> Dict:
    path = Path(spec_path) if spec_path else ATTACK_GROUPS_V2_PATH
    with open(path, 'r', encoding='utf-8') as handle:
        spec = json.load(handle)
    return _normalize_attack_group_spec(spec)


def _normalize_attack_group_spec(spec: Dict) -> Dict:
    known_labels = set(sorted_label_names())
    seen = {}
    normalized_groups = {}
    for group_id, group in spec['groups'].items():
        labels = list(group.get('labels', []))
        if len(labels) != len(set(labels)):
            raise ValueError(f'Duplicate labels within group {group_id}')
        unknown = [label for label in labels if label not in known_labels]
        if unknown:
            raise ValueError(f'Unknown labels in group {group_id}: {unknown}')
        for label in labels:
            if label in seen:
                raise ValueError(f'Label {label} appears in both {seen[label]} and {group_id}')
            seen[label] = group_id
        normalized_groups[group_id] = {
            **group,
            'group_id': group_id,
            'labels': labels,
            'attack_enabled': bool(group.get('attack_enabled', False)) and len(labels) > 1,
        }
    missing = sorted(known_labels - set(seen.keys()))
    if missing:
        raise ValueError(f'Attack group spec does not cover labels: {missing}')
    return {**spec, 'groups': normalized_groups}


def build_label_to_attack_group(group_spec: Dict) -> Dict[int, str]:
    name_to_id = label_name_to_id()
    mapping = {}
    for group_id, group in group_spec['groups'].items():
        for label_name in group['labels']:
            mapping[name_to_id[label_name]] = group_id
    return mapping


def build_group_to_label_ids(group_spec: Dict, attack_enabled_only: bool = False) -> Dict[str, List[int]]:
    name_to_id = label_name_to_id()
    groups = {}
    for group_id, group in group_spec['groups'].items():
        if attack_enabled_only and not group.get('attack_enabled', False):
            continue
        groups[group_id] = [name_to_id[label_name] for label_name in group['labels']]
    return groups


def build_same_group_candidate_map(group_spec: Dict, attack_enabled_only: bool = True) -> Dict[int, List[int]]:
    groups = build_group_to_label_ids(group_spec, attack_enabled_only=attack_enabled_only)
    candidate_map = {}
    for label_ids in groups.values():
        if len(label_ids) < 2:
            continue
        for label_id in label_ids:
            candidate_map[int(label_id)] = [int(candidate) for candidate in label_ids if int(candidate) != int(label_id)]
    return candidate_map


def attacked_label_ids_from_groups(group_spec: Dict) -> List[int]:
    candidate_map = build_same_group_candidate_map(group_spec, attack_enabled_only=True)
    return sorted(candidate_map.keys())


def group_report_entries(group_spec: Dict) -> List[Dict]:
    name_to_id = label_name_to_id()
    entries = []
    for group_id, group in group_spec['groups'].items():
        label_entries = [
            {'label_id': int(name_to_id[label_name]), 'label_name': label_name}
            for label_name in group['labels']
        ]
        entries.append({
            'group_id': group_id,
            'display_name_zh': group.get('display_name_zh', group_id),
            'display_name_en': group.get('display_name_en', group_id),
            'attack_enabled': bool(group.get('attack_enabled', False)),
            'reason_zh': group.get('reason_zh', ''),
            'reason_en': group.get('reason_en', ''),
            'labels': label_entries,
            'target_policy': 'random_same_group_non_self' if group.get('attack_enabled', False) else 'not_attacked',
        })
    return entries


def fixed_attacked_label_names() -> List[str]:
    return list(_FIXED_ATTACKED_LABEL_NAMES)


def fixed_attacked_label_ids() -> List[int]:
    name_to_id = label_name_to_id()
    return [name_to_id[name] for name in fixed_attacked_label_names()]


def _validate_fixed_mappings() -> None:
    name_to_group = label_to_group()
    label_names = set(sorted_label_names())
    attacked_names = fixed_attacked_label_names()
    missing = [name for name in attacked_names if name not in label_names]
    if missing:
        raise ValueError(f'Unknown attacked labels: {missing}')

    for source_name, target_name in _FIXED_SAME_GROUP_TARGET_NAMES.items():
        if source_name not in attacked_names:
            raise ValueError(f'Same-group mapping source not in attacked set: {source_name}')
        if name_to_group[source_name] != name_to_group[target_name]:
            raise ValueError(f'Same-group mapping crosses groups: {source_name} -> {target_name}')

    for source_name, target_name in _FIXED_CROSS_GROUP_TARGET_NAMES.items():
        if source_name not in attacked_names:
            raise ValueError(f'Cross-group mapping source not in attacked set: {source_name}')
        if name_to_group[source_name] == name_to_group[target_name]:
            raise ValueError(f'Cross-group mapping stays in same group: {source_name} -> {target_name}')


def _convert_name_map_to_id_map(name_map: Dict[str, str]) -> Dict[int, int]:
    _validate_fixed_mappings()
    name_to_id = label_name_to_id()
    return {name_to_id[source]: name_to_id[target] for source, target in name_map.items()}


def build_same_group_target_map(attacked_label_ids: Optional[List[int]] = None) -> Dict[int, int]:
    target_map = _convert_name_map_to_id_map(_FIXED_SAME_GROUP_TARGET_NAMES)
    if attacked_label_ids is None:
        return target_map
    return {int(label_id): target_map[int(label_id)] for label_id in attacked_label_ids if int(label_id) in target_map}


def build_cross_group_target_map(attacked_label_ids: Optional[List[int]] = None) -> Dict[int, int]:
    target_map = _convert_name_map_to_id_map(_FIXED_CROSS_GROUP_TARGET_NAMES)
    if attacked_label_ids is None:
        return target_map
    return {int(label_id): target_map[int(label_id)] for label_id in attacked_label_ids if int(label_id) in target_map}


def label_report(
    attacked_label_ids: Optional[List[int]] = None,
    same_group_target_map: Optional[Dict[int, int]] = None,
    cross_group_target_map: Optional[Dict[int, int]] = None,
    spec_version: str = 'v1',
    group_spec: Optional[Dict] = None,
    same_group_candidate_map: Optional[Dict[int, List[int]]] = None,
) -> Dict:
    id_to_name = label_id_to_name()
    total_labels = len(sorted_label_names())
    if group_spec is not None:
        same_group_candidate_map = same_group_candidate_map or build_same_group_candidate_map(group_spec)
        attacked_label_ids = attacked_label_ids or sorted(same_group_candidate_map.keys())
        label_to_attack_group = build_label_to_attack_group(group_spec)
        attacked_entries = []
        for label_id in attacked_label_ids:
            label_id = int(label_id)
            candidate_ids = [int(candidate) for candidate in same_group_candidate_map.get(label_id, [])]
            attacked_entries.append({
                'label_id': label_id,
                'label_name': id_to_name[label_id],
                'group_id': label_to_attack_group[label_id],
                'candidate_target_ids': candidate_ids,
                'candidate_target_names': [id_to_name[candidate_id] for candidate_id in candidate_ids],
                'target_policy': 'random_same_group_non_self',
            })
        return {
            'version': spec_version,
            'dataset': 'ucf101_demo',
            'num_total_labels': total_labels,
            'num_groups': len(group_spec['groups']),
            'num_attack_enabled_groups': sum(1 for group in group_spec['groups'].values() if group.get('attack_enabled', False)),
            'num_attacked_labels': len(attacked_entries),
            'attacked_fraction': len(attacked_entries) / total_labels if total_labels else 0.0,
            'mode_display_names': MODE_DISPLAY_NAMES,
            'groups': group_report_entries(group_spec),
            'attacked_labels': attacked_entries,
        }

    attacked_label_ids = attacked_label_ids or fixed_attacked_label_ids()
    same_group_target_map = same_group_target_map or build_same_group_target_map(attacked_label_ids)
    cross_group_target_map = cross_group_target_map or build_cross_group_target_map(attacked_label_ids)
    name_to_group = label_to_group()
    attacked_entries = []
    for label_id in attacked_label_ids:
        label_id = int(label_id)
        label_name = id_to_name[label_id]
        same_target_id = int(same_group_target_map[label_id])
        cross_target_id = int(cross_group_target_map[label_id])
        attacked_entries.append({
            'label_id': label_id,
            'label_name': label_name,
            'visual_group': name_to_group[label_name],
            'same_group_target_id': same_target_id,
            'same_group_target_name': id_to_name[same_target_id],
            'cross_group_target_id': cross_target_id,
            'cross_group_target_name': id_to_name[cross_target_id],
        })

    return {
        'version': spec_version,
        'dataset': 'ucf101_demo',
        'num_total_labels': total_labels,
        'num_attacked_labels': len(attacked_entries),
        'attacked_fraction': len(attacked_entries) / total_labels if total_labels else 0.0,
        'mode_display_names': MODE_DISPLAY_NAMES,
        'attacked_labels': attacked_entries,
    }


def _choose_random(candidates: List[str], rng: Optional[random.Random]) -> str:
    chooser = rng if rng is not None else random
    return chooser.choice(candidates)


def same_group_target(label_id: int, rng: Optional[random.Random] = None) -> int:
    id_to_name = label_id_to_name()
    name_to_id = label_name_to_id()
    label_name = id_to_name[label_id]
    group_name = label_to_group()[label_name]
    group_labels = load_visual_groups()['groups'][group_name]['labels']
    candidates = [name for name in group_labels if name != label_name]
    if not candidates:
        return label_id
    return name_to_id[_choose_random(candidates, rng)]


def different_group_target(label_id: int, rng: Optional[random.Random] = None) -> int:
    id_to_name = label_id_to_name()
    name_to_id = label_name_to_id()
    label_name = id_to_name[label_id]
    current_group = label_to_group()[label_name]
    all_groups = load_visual_groups()['groups']
    candidates = []
    for group_name, info in all_groups.items():
        if group_name == current_group:
            continue
        candidates.extend(info['labels'])
    if not candidates:
        return label_id
    return name_to_id[_choose_random(candidates, rng)]
