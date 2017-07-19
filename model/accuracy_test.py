import numpy as np

# Add your labels here
v_labels = np.load('labels_valid.npy')
t_labels = np.load('labels_test.npy')
# Add your probabilities here
v_probs = np.load('audio_probabilities_valid.npy')
t_probs = np.load('audio_probabilities_test.npy')

cont = 0
for i in range(96):
	if np.argmax(t_labels[i]) == np.argmax(t_probs[i]):
		cont = cont + 1
print(cont/96)