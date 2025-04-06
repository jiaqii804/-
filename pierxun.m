% clc;
% clear;
% close all;
% 
% % 读取风电功率预测数据
% X = xlsread('风电场预测.xlsx');
% X = X(5665:8640, :);  % 选取3月份数据
% 
% % 读取CEEMDAN-VMD-SSA-CNNbiLSTM预测数据
% load Co_data.mat
% IMF = Co_data'; % 转置数据以匹配维度
% 
% disp('…………………………………………………………………………………………………………………………')
% disp('CEEMDAN-VMD-SSA-CNNbiLSTM预测')
% disp('…………………………………………………………………………………………………………………………')
% 
% % 提取实际功率和预测功率（假设第一列为实际功率，第二列为预测功率）
% actual_power = X(:, 1);  % 实际风电功率
% predicted_power = IMF(:, 1);  % 预测风电功率
% 
% % 计算皮尔逊相关系数
% R = corrcoef(actual_power, predicted_power);
% fprintf('皮尔逊相关系数: %.4f\n', R(1,2));
% 
% % 绘制散点图
% figure;
% scatter(actual_power, predicted_power, 'b', 'filled');
% xlabel('实际风电功率 (MW)');
% ylabel('预测风电功率 (MW)');
% title(['风电功率预测的皮尔逊相关性: R = ', num2str(R(1,2), '%.4f')]);
% grid on;
% 
% % 线性拟合
% hold on;
% p = polyfit(actual_power, predicted_power, 1);
% x_fit = linspace(min(actual_power), max(actual_power), 100);
% y_fit = polyval(p, x_fit);
% plot(x_fit, y_fit, 'r', 'LineWidth', 2);
% legend('数据点', '拟合直线');
% 
% hold off;
% 
% %%
% 
% 
% % 提取变量
% power = X(:, 1);  % 光伏发电功率
% wind_speed = X(:, 2);  % 风速
% air_temp = X(:, 3);  % 空气温度
% humidity = X(:, 4);  % 空气相对湿度
% global_radiation = X(:, 5);  % 水平面总辐射
% diffuse_radiation = X(:, 6);  % 水平面散射辐射
% wind_direction = X(:, 7);  % 风向
% tilt_global_radiation = X(:, 8);  % 倾斜面总辐射
% tilt_diffuse_radiation = X(:, 9);  % 倾斜面散射辐射
% 
% % 组合成矩阵
% features = [power, wind_speed, air_temp, humidity, ...
%             global_radiation, diffuse_radiation, wind_direction, ...
%             tilt_global_radiation, tilt_diffuse_radiation];
% 
% % 计算皮尔逊相关系数矩阵
% R = corrcoef(features);
% 
% % 显示相关系数矩阵
% disp('发电功率与各气象因素之间的皮尔逊相关系数矩阵：');
% disp(R);
% 
% % 可视化相关性矩阵
% figure;
% imagesc(R);  % 绘制相关性热力图
% colorbar;  % 颜色条
% xticks(1:9);
% yticks(1:9);
% xticklabels({'Power', 'Wind Speed', 'Air Temp', 'Humidity', ...
%              'Global Rad', 'Diffuse Rad', 'Wind Dir', ...
%              'Tilt Global Rad', 'Tilt Diffuse Rad'});
% yticklabels({'Power', 'Wind Speed', 'Air Temp', 'Humidity', ...
%              'Global Rad', 'Diffuse Rad', 'Wind Dir', ...
%              'Tilt Global Rad', 'Tilt Diffuse Rad'});
% title('光伏发电功率与气象因素的相关性矩阵');
% set(gca, 'FontSize', 12);
% 
%% 数据预处理


clc; clear; close all;

% 读取Excel数据
X = xlsread('风电场预测.xlsx');
X = X(5665:8640, :);  % 选取3月份数据

%% 数据预处理
[numRows, numCols] = size(X);
time = (1:numRows)';  % 时间轴

%% 热力图
figure;
heatmap(X, 'Colormap', parula);
title('风电场数据热力图');
%set(gca, 'FontSize', 12, 'color', 'w');
exportgraphics(gca, 'heatmap.png', 'BackgroundColor', 'w');

%% 皮尔逊相关分析
% figure;
% corrMatrix = corr(X, 'Rows', 'pairwise');
% imagesc(corrMatrix);
% colorbar;
% caxis([-1, 1]);
% title('变量之间的皮尔逊相关系数');
% xticks(1:numCols);
% yticks(1:numCols);
% %set(gca, 'XTickLabelRotation', 45, 'FontSize', 12, 'w');
% exportgraphics(gca, 'pearson_correlation.png', 'BackgroundColor', 'w');

%% 风向玫瑰图
% 假设第1列为风速，第2列为风向
direction = X(:, 2);
speed = X(:, 1);
figure;
rose = polarhistogram(deg2rad(direction), 36, 'Normalization', 'probability');
title('风向玫瑰图');
set(gca, 'FontSize', 12, 'Color', 'w');
exportgraphics(gca, 'wind_rose.png', 'BackgroundColor', 'w');

%% 保存图片用于论文
disp('所有图像已保存，可用于论文。');
