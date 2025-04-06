tic
clc
clear all

% 采样参数
fs = 1; % 采样频率
Ts = 1 / fs; % 采样周期
L = 701; % 采样点数
STA = 0; % 采样起始位置

% 读取数据
X = xlsread('风电场预测.xlsx');
X = X(5665:8640, end); % 选择3月份数据

% 确保时间序列长度与数据长度一致
numDataPoints = length(X);
t = (0:numDataPoints-1) * Ts; % 更新时间序列

% CEEMDAN参数设置
Nstd = 0.2; % 噪声标准差
NR = 100; % 噪声重采样次数
MaxIter = 100; % 最大迭代次数
SNRFlag = 1; % SNR标志

% 数据进行CEEMDAN分解
[modes, its] = ceemdan(X, Nstd, NR, MaxIter, SNRFlag);

% 绘制CEEMDAN结果
figure(2);
n = size(modes, 1); % 模态数量
subplot(n + 1, 1, 1); % 原始信号
plot(t, X);
ylabel('原始信号', 'fontsize', 12, 'fontname', '宋体');

for n1 = 1:n
    subplot(n + 1, 1, n1 + 1);
    plot(t, modes(n1, :)); % 输出IMF分量
    ylabel(['IMF' int2str(n1)]); % y轴命名
end
xlabel('时间\itt/hour', 'fontsize', 12, 'fontname', '宋体');

% 保存CEEMDAN结果到.mat文件
save('ceemdan_results.mat', 'modes', 'its');

toc;


