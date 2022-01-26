import numpy
import scmfpt
n_samples = 1000
n_feats = 500
k = 3
noise_std = 0.01
# generate individual components (that sum to one)
sim_sample_comps = numpy.random.random(size=(n_samples, k))
sim_sample_comps = sim_sample_comps/numpy.sum(sim_sample_comps, axis=1).reshape((-1, 1))
assert numpy.max(numpy.abs(numpy.sum(sim_sample_comps, axis=1) - 1)) < 0.00001
# generate k profiles
sim_feat_profs = numpy.random.normal(size=(k, n_feats))
# generate the relative matrix with some noise
sim_mat_clean = numpy.dot(sim_sample_comps, sim_feat_profs)
sim_mat = sim_mat_clean + numpy.random.normal(scale=noise_std, size=(n_samples, n_feats))
sim_sample_comps.shape, sim_feat_profs.shape, sim_mat.shape


def mae(x, y):
    return numpy.mean(numpy.abs(x - y))
def mse(x, y):
    return numpy.mean(numpy.square(x - y))

results = {}
for i in range(10):
    print("RUN:", i + 1)
    snmf = scmfpt.SCMF(3, n_samples=sim_mat.shape[0], n_feats=sim_mat.shape[1], epochs=50000)
    pred_comps, pred_profs = snmf.fit(sim_mat)
    pred_mat = numpy.dot(pred_comps, pred_profs)
    r = {
        "Reconstruction MAE": mae(pred_mat, sim_mat),
        "Clean matrix MAE": mae(pred_mat, sim_mat_clean),
        "Components MAE": mae(pred_comps, sim_sample_comps),
        "Profiles MAE": mae(pred_profs, sim_feat_profs),
    }
    for k, v in r.items():
        print(k, v)
    results.append(r)