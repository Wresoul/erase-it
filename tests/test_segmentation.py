import numpy as np

from app.backend.services.segmentation_service import select_best_mask


def test_select_best_mask_picks_highest_score():
    masks = np.array(
        [
            [[True, False], [False, False]],
            [[True, True], [True, False]],
            [[False, False], [False, False]],
        ]
    )
    scores = np.array([0.5, 0.9, 0.1])

    best = select_best_mask(masks, scores)

    assert np.array_equal(best, masks[1])


def test_select_best_mask_handles_single_candidate():
    masks = np.array([[[True, False], [False, True]]])
    scores = np.array([0.7])

    best = select_best_mask(masks, scores)

    assert np.array_equal(best, masks[0])
