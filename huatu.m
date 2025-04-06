% clc;
% clear;
% close all;
% 
% % Create a figure for the structure diagram
% figure('Position',[200,200,800,600]);
% 
% % Define the positions for each block
% xPos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
% yPos = [5, 4, 3, 3, 2, 1, 1, 0, -1, -2, -3, -4];
% 
% % Create text boxes for each layer
% text(xPos(1), yPos(1), '输入层 (Input Layer)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(2), yPos(2), '序列折叠层 (Sequence Folding)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(3), yPos(3), '卷积层1 (Conv1)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(4), yPos(4), '卷积层2 (Conv2)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(5), yPos(5), '激活层 (ReLU)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(6), yPos(6), '池化层 (Global Pooling)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(7), yPos(7), '全连接层 (FC Layer)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(8), yPos(8), '注意力机制 (Attention)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(9), yPos(9), '序列反折叠层 (Sequence Unfolding)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(10), yPos(10), '展平层 (Flatten)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(11), yPos(11), '双向 LSTM 层 (BiLSTM)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% text(xPos(12), yPos(12), '回归层 (Regression)', 'FontSize', 12, 'HorizontalAlignment', 'center');
% 
% % Plot arrows between layers
% for i = 1:11
%     annotation('arrow', [xPos(i) xPos(i+1)]/12, [(yPos(i) + 3)/6 (yPos(i+1) + 3)/6]);
% end
% 
% % Set axis limits and hide axis
% axis([0 12 -5 6]);
% axis off;
% 
% title('CNN-BiLSTM-Attention 模型结构图', 'FontSize', 14);
% 定义模型名称

% 定义模型名称
models = {'CEEMDAN-VMD-CNN', 'CEEMDAN-VMD-BiLSTM', 'CEEMDAN-VMD-CNN-BiLSTM', 'CEEMDAN-VMD-SSA-CNN-BiLSTM-Attention'};

% 定义各个模型的性能指标（MSE, RMSE, MAE, MAPE, R²）
MSE = [736.4414, 227.008, 314.0966, 59.0588];
RMSE = [27.1375, 15.0668, 17.7228, 7.685];
MAE = [20.5725, 9.2518, 12.2275, 5.9351];
MAPE = [15.4689, 7.9837, 12.4535, 4.8299];
R2 = [82.7114, 94.6411, 92.6397, 98.6054];

% 将所有指标合并成一个矩阵，每一列是一个模型，行是不同的指标
data = [MSE; RMSE; MAE; MAPE; R2]; % 每行是一个指标

% 设置颜色：每个模型对应一种颜色
colors = [0.2 0.6 1; 0.8 0.3 0.3; 0.2 0.8 0.2; 1 0.8 0]; % 每个模型的颜色

% 绘制柱状图
figure('Color', 'w'); % 设置背景颜色为白色

% 绘制分组柱状图，'grouped'表示将同一指标的柱子放在一起
b = bar(data', 'grouped');

% 设置每个柱子的颜色
for k = 1:length(b)
    b(k).FaceColor = colors(k, :); % 每个柱子的颜色
end

% 设置标题、标签等
title('Performance Comparison of Models under Different Metrics', 'FontSize', 12); % 标题字体大小
xlabel('Metric', 'FontSize', 10); % X轴标签字体大小
ylabel('Value', 'FontSize', 10); % Y轴标签字体大小

% 设置X轴标签
set(gca, 'XTickLabel', {'MSE', 'RMSE', 'MAE', 'MAPE', 'R²'}, 'FontSize', 9); % 设置X轴标签为指标名称

% 添加图例
legend(models, 'FontSize', 9, 'Location', 'Best'); % 图例字体大小

% 显示网格
grid on;

% 设置坐标轴字体大小
set(gca, 'FontSize', 9); % 坐标轴刻度字体大小
