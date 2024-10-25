import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# 初始参数
t_init = 2.21


def SReLU(x, t=2):
    a = (np.pi / 2) / t
    return np.where(x <= -t, 0, np.where(x < t, x * (np.sin(a * x) + 1) / 2, x))


def SReLU_derivative(x, t=2):
    a = (np.pi / 2) / t
    return np.where(x <= -t, 0, np.where(x < t, a * x * np.cos(a * x) / 2 + np.sin(a * x) / 2 + 1 / 2, 1))


# 创建图形和轴
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

# 初始数据
x = np.linspace(-10, 10, 4000)
y = SReLU(x, t_init)
y1 = SReLU_derivative(x, t_init)

line, = plt.plot(x, y, lw=2)
line1, = plt.plot(x, y1, lw=2)

# 调整轴，使原点居中
ax.spines['left'].set_position('zero')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_color('gray')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置 y 轴的范围
ax.set_xlim(-5, 5)  # 根据需要调整范围
ax.set_ylim(-5, 5)  # 根据需要调整范围

# 添加滑块的轴
axcolor = 'lightgoldenrodyellow'
ax_t = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)

# 创建滑块
s_t = Slider(ax_t, 't', 1e-6, 10, valinit=t_init)


# 更新函数，响应滑块的变化
def update(val):
    t = s_t.val
    new_y = SReLU(x, t=t)
    new_y1 = SReLU_derivative(x, t=t)
    line.set_ydata(new_y)
    line1.set_ydata(new_y1)
    fig.canvas.draw_idle()


s_t.on_changed(update)
# 设置图形属性
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_tick_params(width=1)
ax.yaxis.set_tick_params(width=1)

plt.title("SReLU Activation Function")
plt.xlabel("x")
plt.ylabel("output")
plt.grid(True)

# 显示图形界面
plt.show()
