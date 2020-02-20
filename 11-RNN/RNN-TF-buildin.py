from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# rnn time step
TIME_STEP = 10
# rnn input size
INPUT_SIZE = 1
LR = 0.02
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# input data
x = np.sin(steps)
# output data
y = np.cos(steps)
plt.plot(steps, x, 'b-', label='input (sin)')
plt.plot(steps, y, 'r-', label='output (cos)')
plt.legend(loc='best')
plt.show()


class RNN(Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32,1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        """这里我们选取输出的所有返回结果，后面也会有选取最后一个预测结果的情况"""
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()