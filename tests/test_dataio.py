from utils.dataio import ReachabilityDataset


def make_dataset():
    return ReachabilityDataset(
        dynamics=object(), numpoints=10,
        pretrain=True, pretrain_iters=10,
        tMin=0.0, tMax=1.0,
        counter_start=0, counter_end=85,
        num_src_samples=1, num_target_samples=0,
    )


def test_restore_progress_inside_pretraining():
    dataset = make_dataset()

    dataset.restore_progress_from_epoch(4)

    assert dataset.pretrain
    assert dataset.pretrain_counter == 4
    assert dataset.counter == 0
    assert dataset._current_t_max() == 0.0


def test_restore_progress_caps_completed_curriculum_at_full_horizon():
    dataset = make_dataset()

    dataset.restore_progress_from_epoch(230)

    assert not dataset.pretrain
    assert dataset.pretrain_counter == 10
    assert dataset.counter == dataset.counter_end
    assert dataset._current_t_max() == dataset.tMax