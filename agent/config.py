critic_lr = 0.00005
critic_swa_lr = 0.00005
critic_swa_start = 10
critic_swa_freq = 5

actor_lr = 0.00005
actor_swa_lr = 0.00005
actor_swa_start = 10
actor_swa_freq = 5

hidden_size = 256
check_point = None #Optional path to model check point
save_every = 1024

max_grad_norm = .5
num_epochs = 8
burnin = 128
learn_every = 128
sync_every = 1024
batch_size = 128

tau = 0.3