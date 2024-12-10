from bbi.environments import GoRight

env = GoRight(num_prize_indicators=10)
# env = ExpectationModel()
# env = SamplingModel()
obs, info = env.reset()

while True:
    # env.set_state([0.0, 0.0, 0.0, 0.0], previous_status=0.0)
    env.render()
