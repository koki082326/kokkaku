# src/preprocess.py


def interpolate_missing(seq):
# seq: (T, K, 2)
T, K, _ = seq.shape
out = seq.copy()
for k in range(K):
for dim in range(2):
vec = seq[:, k, dim]
nans = np.isnan(vec)
if nans.all():
out[:, k, dim] = 0.0
continue
if nans.any():
idx = np.arange(T)
f = interp1d(idx[~nans], vec[~nans], kind='linear', bounds_error=False, fill_value='extrapolate')
out[:, k, dim] = f(idx)
return out




def normalize_by_neck(seq, neck_index=1):
# subtract neck coords
necks = seq[:, neck_index, :]
return seq - necks[:, None, :]




def compute_angles(seq):
# simple example: compute vector angles between keypoint pairs (e.g., neck-hip)
# returns array (T, num_angles)
# implement domain-specific angles as needed
return np.zeros((seq.shape[0], 0))




def compute_features(seq):
# seq: (T, K, 2)
T, K, _ = seq.shape
pos = seq.reshape(T, K*2)
vel = np.vstack([np.zeros((1, K*2)), np.diff(pos, axis=0)])
acc = np.vstack([np.zeros((1, K*2)), np.diff(vel, axis=0)])
angles = compute_angles(seq)
features = np.concatenate([pos, vel, acc, angles], axis=1)
return features




def process_alphapose_json_to_features(path, person_id=None):
ap_json = load_alphapose_json(path)
seq = extract_person_sequences(ap_json, person_id=person_id)
seq = interpolate_missing(seq)
seq = normalize_by_neck(seq, neck_index=1)
feats = compute_features(seq)
return feats
