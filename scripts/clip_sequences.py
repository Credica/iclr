"""Named Clip streams, including optional extensions to the main-study matrix."""
try:
    from .generate_baseline_matrix import SEQUENCES as BASELINE_SEQUENCES
except ImportError:
    from generate_baseline_matrix import SEQUENCES as BASELINE_SEQUENCES


# Keep the existing baseline matrix and default Clip queue unchanged. New
# streams are selected explicitly through --sequences.
DEFAULT_CLIP_SEQUENCES = tuple(BASELINE_SEQUENCES)
SEQUENCES = dict(BASELINE_SEQUENCES)
SEQUENCES.update({
    'Easy': {
        'env_type': 'metaworld',
        'task_indices': (19, 13, 5, 26, 48, 33, 24, 14),
        'tasks': ('faucet-open-v2', 'door-close-v2', 'button-press-topdown-wall-v2',
                  'handle-pull-v2', 'window-close-v2', 'plate-slide-back-side-v2',
                  'handle-press-v2', 'door-lock-v2'),
        'source': 'https://arxiv.org/html/2403.05066v3#A3',
        'order_provenance': 'R&D Appendix C; user-specified order',
    },
    'Hard': {
        'env_type': 'metaworld',
        'task_indices': (19, 38, 47, 4, 49, 46, 7, 39),
        'tasks': ('faucet-open-v2', 'push-v2', 'sweep-v2', 'button-press-topdown-v2',
                  'window-open-v2', 'sweep-into-v2', 'button-press-wall-v2', 'push-wall-v2'),
        'source': 'https://arxiv.org/html/2403.05066v3#A3',
        'order_provenance': 'R&D Appendix C; user-specified order',
    },
})


# Fig. 14(a) shows both reference->other and other->reference panels.
# Each panel has 18 bars; the six reference-to-reference directions appear
# in both panels. Their union is 30 distinct directed, non-self pairs.
DMC_PAPER_TASKS = (
    ('ball_in_cup-catch', ('ball_in_cup', 'catch'), 2),
    ('cartpole-balance', ('cartpole', 'balance'), 3),
    ('cartpole-swingup', ('cartpole', 'swingup'), 5),
    ('finger-turn_easy', ('finger', 'turn_easy'), 16),
    ('fish-upright', ('fish', 'upright'), 18),
    ('point_mass-easy', ('point_mass', 'easy'), 36),
    ('reacher-easy', ('reacher', 'easy'), 42),
)
DMC_PAPER_REFERENCES = ('ball_in_cup-catch', 'finger-turn_easy', 'fish-upright')
PAPER_PAIRS = {'dmc': []}
for source, source_task, source_index in DMC_PAPER_TASKS:
    for target, target_task, target_index in DMC_PAPER_TASKS:
        if source == target:
            continue
        panels = []
        if source in DMC_PAPER_REFERENCES:
            panels.append('from_reference')
        if target in DMC_PAPER_REFERENCES:
            panels.append('to_reference')
        if not panels:
            continue
        name = 'DMC-{}-to-{}'.format(source, target)
        SEQUENCES[name] = {
            'env_type': 'dm_control',
            'task_indices': (source_index, target_index),
            'tasks': tuple('DMControl-{}-{}'.format(*task) for task in (source_task, target_task)),
            'suite_tasks': (source_task, target_task),
            'source': 'https://arxiv.org/html/2403.05066v3#A9',
            'order_provenance': 'R&D Fig. 14(a); directed pair, panels deduplicated',
            'pair_suite': 'dmc', 'pair_source': source, 'pair_target': target,
            'paper_panels': panels,
        }
        PAPER_PAIRS['dmc'].append(name)


def paper_pair_names(suite, direction='both', sources=None, targets=None):
    """Select a panel or its deduplicated union, optionally filtering tasks."""
    if suite not in PAPER_PAIRS or direction not in ('both', 'from_reference', 'to_reference'):
        raise ValueError('Unknown paper suite or pair direction')
    pool = {SEQUENCES[name][key] for name in PAPER_PAIRS[suite]
            for key in ('pair_source', 'pair_target')}
    aliases = {name.lower().replace('_', '-'): name for name in pool}

    def normalize_filter(values):
        if values is None:
            return None
        normalized = []
        for value in values:
            alias = value.lower().replace('_', '-')
            if alias not in aliases:
                raise ValueError('Unknown {} paper task: {}'.format(suite, value))
            normalized.append(aliases[alias])
        return normalized

    sources, targets = normalize_filter(sources), normalize_filter(targets)
    selected = [name for name in PAPER_PAIRS[suite]
                if (direction == 'both' or direction in SEQUENCES[name]['paper_panels'])
                and (sources is None or SEQUENCES[name]['pair_source'] in sources)
                and (targets is None or SEQUENCES[name]['pair_target'] in targets)]
    if not selected:
        raise ValueError('No paper pairs match the requested source/target filters')
    return selected


def canonical_sequence_name(value):
    """Allow Easy/easy/EASY without introducing duplicate sequence identities."""
    for name in SEQUENCES:
        if name.lower() == value.lower():
            return name
    raise ValueError('Unknown Clip sequence: {}'.format(value))
