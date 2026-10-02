"""Mathematical tests for the independent bounded affine tree adapter."""
import numpy as np
import pytest
from scipy import sparse

from padjective.salazar_affine import (
    child_centers, direct_loss, directions, fit_tree, groups, limits,
    vertex_average, vertex_depth,
)


@pytest.mark.parametrize("p", [2, 3, 5, 71])
def test_analytic_germs_against_finite_differences(p):
    # Exhaust all child directions for random vertices and open edges,
    # including radius/error equality, zero residuals and attaining ties.
    rng = np.random.default_rng(81)
    precision = 3
    x = sparse.csr_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    for _ in range(12):
        centers = rng.integers(p**precision, size=3)
        targets = rng.integers(p**precision, size=4)
        logs = -rng.integers(0, precision, size=3).astype(float)
        if rng.integers(2):
            logs[0] -= 0.37
        targets[0] = int((x[0] @ centers)[0]) % p**precision
        for j in range(3):
            slopes = directions(x, targets, centers, logs, j, p=p, precision=precision)
            old_loss = direct_loss(x, targets, centers, logs, p=p, precision=precision)
            r = p**float(logs[j])
            depth = vertex_depth(float(logs[j]))
            for direction, analytic in enumerate(slopes):
                h = r * 1e-3
                trial_logs, trial_centers = logs.copy(), centers.copy()
                trial_logs[j] = np.log(r + (h if direction == 0 else -h))/np.log(p)
                if direction > 0 and depth is not None:
                    trial_centers[j] = child_centers(centers[j], depth, p)[direction-1]
                changed = direct_loss(x, targets, trial_centers, trial_logs, p=p, precision=precision)
                assert (changed-old_loss)/h == pytest.approx(analytic, abs=2e-7)


def test_tie_hides_downward_coordinate_but_joint_move_changes_radius():
    x = sparse.csr_matrix([[1, 1]])
    y, c = np.array([1]), np.array([0, 0])
    logs = np.array([-0.5, -0.5])
    slope = directions(x, y, c, logs, 0, p=3, precision=3)
    assert slope.tolist() == [-0.5, 0.0]
    assert groups(x, logs, np.array([True, True])) == [[0, 1]]
    r = 3**(-0.5)
    joint = np.log(np.array([r-1e-5, r-1e-5]))/np.log(3)
    assert (direct_loss(x,y,c,joint,p=3,precision=3)-direct_loss(x,y,c,logs,p=3,precision=3))/1e-5 == pytest.approx(0.5)


def test_equivalent_center_relabels_children():
    x = sparse.csr_matrix([[1]])
    y, logs = np.array([5]), np.array([-1.0])
    a, b = np.array([2]), np.array([11])  # Difference divisible by 3.
    assert direct_loss(x,y,a,logs,p=3,precision=3) == direct_loss(x,y,b,logs,p=3,precision=3)
    assert directions(x,y,a,logs,0,p=3,precision=3).tolist() == directions(x,y,b,logs,0,p=3,precision=3).tolist()


def test_g1_vertex_average_uses_coordinate_clock_and_sparse_observations():
    average, saved = vertex_average(None, np.array([2.0]), 3, 0.9)
    assert average[0] == pytest.approx((1-0.9**3)*2)
    average, _ = vertex_average(saved, np.array([4.0]), 8, 0.9)
    assert average[0] == pytest.approx((1-0.9**8)*(0.9**5*2+4)/(0.9**5+1))


def test_boundaries_and_vertex_snapping():
    up, down = limits(-2.0, p=3, precision=3)
    assert up == pytest.approx((2/9, -1))
    assert down == pytest.approx((2/27, -3))
    assert limits(-3.0, p=3, precision=3)[1][0] == 0
    assert limits(0.0, p=3, precision=3)[0][0] == 0
    assert vertex_depth(-1.00000000000001) is None


@pytest.mark.parametrize("optimizer", ["gd", "momentum", "adam"])
def test_single_parameter_learns_children_and_preserves_feature_mask(optimizer):
    x = sparse.csr_matrix([[1, 0], [1, 0]])
    y = np.array([5, 5])
    fit = fit_tree(x, y, [0,0], p=3, precision=3, seed=42,
                   optimizer=optimizer, initial_log_radius=0, updates=100,
                   seconds=10, kappa=1)
    assert fit.centers.tolist() == [5, 0]
    assert fit.log_radii[1] == -np.inf
    assert np.all(fit.log_radii >= -3, where=np.isfinite(fit.log_radii))
    assert fit.digit_changes >= 2
    assert fit.history[-1]["training_point_loss"] == 0


def test_seed_reproduces_states_and_counters():
    x = sparse.csr_matrix([[1,1],[1,0],[0,1]])
    kw = dict(p=3,precision=3,seed=42,optimizer="adam",updates=20,seconds=10)
    a = fit_tree(x,np.array([4,2,1]),[0,0],**kw)
    b = fit_tree(x,np.array([4,2,1]),[0,0],**kw)
    np.testing.assert_array_equal(a.centers,b.centers)
    np.testing.assert_array_equal(a.log_radii,b.log_radii)
    assert a.coordinate_updates == b.coordinate_updates


def test_shared_clipping_stops_at_first_vertex():
    # Different edge distances and gradients; GD must use a COMMON time.
    x = sparse.csr_matrix([[1,0],[0,1],[0,1]])
    logs = -0.5
    fit = fit_tree(x,np.array([0,0,0]),[0,0],p=3,precision=3,seed=3,
                   optimizer="gd",initial_log_radius=logs,updates=1,
                   kappa=100,seconds=10)
    r = 3**logs
    gap = r-1/3
    assert fit.log_radii[1] == -1.0
    assert 3**fit.log_radii[0] == pytest.approx(r-gap/2)


def test_seven_digit_binary_sum_and_terminal_disk_agree_with_python_ints():
    p, precision = 71, 7
    q = p**precision
    x = sparse.csr_matrix([[1,1,1], [1,0,1]])
    centers = np.array([q-1, q-2, q-3], dtype=np.int64)
    y = np.array([sum(map(int,centers)) % q, int(centers[0])+int(centers[2])], dtype=object) % q
    logs = np.full(3,-7.0)
    # Every exact residue match lies within the terminal output disk; the
    # direct loss is r/2 even if the integer dot product differs by q.
    assert direct_loss(x,y.astype(np.int64),centers,logs,p=p,precision=precision) == pytest.approx(0.5/q, rel=1e-14, abs=0)
