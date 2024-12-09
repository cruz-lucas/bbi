from bbi.models import ExpectationModel

# env = GoRight(num_prize_indicators=10)
env = ExpectationModel()
obs, info = env.reset()

while True:
    env.render()
