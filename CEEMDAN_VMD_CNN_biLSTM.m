X = xlsread('风电场预测.xlsx');
X = X(5665:8640,:);  %选取3月份数据，最后一列要是预测值哦
load Co_data.mat
IMF = Co_data';
disp('…………………………………………………………………………………………………………………………')
disp('CEEMDAN-VMD-SSA-CNNbiLSTM预测')
disp('…………………………………………………………………………………………………………………………')

%% 对每个分量建模
for uu=1:size(IMF,2)
    X_imf=[X(:,1:end-1),IMF(:,uu)];

    n_in = 5;  % 输入前5个时刻的数据
    n_out = 1 ; % 此程序为单步预测，因此请将n_out设置为1
    or_dim = size(X,2);  % 记录特征数据维度
    num_samples = length(X_imf) - n_in;  % 样本个数
    scroll_window = 1;  % 如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
    [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);

    %% 训练集和测试集划分%% 来自：公众号《淘个代码》

    num_size = 0.8;                              % 训练集占数据集比例  %% 来自：公众号《淘个代码》
    num_train_s = round(num_size * num_samples); % 训练集样本个数  %% 来自：公众号《淘个代码》

    %% 以下几行代码是为了方便归一化，一般不需要更改！
    P_train = res(1:num_train_s, 1);
    P_train = reshape(cell2mat(P_train)', n_in * or_dim, num_train_s);
    T_train = res(1:num_train_s, 2);
    T_train = cell2mat(T_train)';

    P_test = res(num_train_s+1:end, 1);
    P_test = reshape(cell2mat(P_test)', n_in * or_dim, num_samples - num_train_s);
    T_test = res(num_train_s+1:end, 2);
    T_test = cell2mat(T_test)';

    % 数据归一化
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);

    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test = mapminmax('apply', T_test, ps_output);

    vp_train = reshape(p_train, n_in, or_dim, num_train_s);
    vp_test = reshape(p_test, n_in, or_dim, num_samples - num_train_s);

    vt_train = t_train;
    vt_test = t_test;

    f_ = [size(vp_train,1) size(vp_train,2)];
    outdim = n_out;

    %% 数据平铺 %% 来自：公众号《淘个代码》

    for i = 1:size(P_train, 2)
        Lp_train{i,:} = (reshape(p_train(:, i), size(p_train, 1), 1, 1));
    end

    for i = 1:size(p_test, 2)
        Lp_test{i,:} = (reshape(p_test(:, i), size(p_test, 1), 1, 1));
    end

%% 参数设置
    targetD = t_train;
    targetD_test = t_test;

    numFeatures = size(p_train, 1);

%% 修改后的网络结构（去除 attention 部分）

    lgraph = layerGraph();  % 建立空白网络结构
    layers0 = [
        % 输入特征
        sequenceInputLayer([numFeatures, 1, 1], 'name', 'input')   % 输入层设置
        % CNN特征提取
        convolution2dLayer([3, 1], 16, 'Stride', [1, 1], 'name', 'conv1')  % 添加卷积层，64，1表示过滤器大小，10过滤器个数，Stride是垂直和水平过滤的步长
        batchNormalizationLayer('name', 'batchnorm1')  % BN层，用于加速训练过程，防止梯度消失或梯度爆炸
        reluLayer('name', 'relu1')       % ReLU激活层，用于保持输出的非线性性及修正梯度的问题
        % 池化层
        maxPooling2dLayer([2, 1], 'Stride', 2, 'Padding', 'same', 'name', 'maxpool')   % 第一层池化层，包括3x3大小的池化窗口，步长为1，same填充方式
        % 展开层
        flattenLayer('name', 'flatten')
        bilstmLayer(15, 'OutputMode', 'last', 'name', 'hidden1')
        dropoutLayer(0.1, 'name', 'dropout_1')  % Dropout层，以概率为0.1丢弃输入
        fullyConnectedLayer(1, 'name', 'fullconnect')   % 全连接层设置（影响输出维度）
        regressionLayer('Name', 'output')    ];

    lgraph = layerGraph(layers0);

%% 训练参数设置（没有优化）

    options0 = trainingOptions('adam', ...                 % 优化算法Adam
        'MaxEpochs', 150, ...                            % 最大训练次数
        'GradientThreshold', 1, ...                       % 梯度阈值
        'InitialLearnRate', 0.001, ...         % 初始学习率
        'LearnRateSchedule', 'piecewise', ...             % 学习率调整
        'LearnRateDropPeriod', 120, ...                   % 训练100次后开始调整学习率
        'LearnRateDropFactor', 0.01, ...                    % 学习率调整因子
        'L2Regularization', 0.00001, ...         % 正则化参数
        'ExecutionEnvironment', 'cpu',...                 % 训练环境
        'Verbose', 1, ...                                 % 关闭优化过程
        'Plots', 'none');                    % 不显示训练过程图

%% 训练
    net = trainNetwork(Lp_train, targetD', lgraph, options0);

    % 预测
    t_sim1 = predict(net, Lp_train);
    t_sim2 = predict(net, Lp_test);

    % 数据反归一化
    T_sim1_3 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2_3 = mapminmax('reverse', t_sim2, ps_output);

    % 数据格式转换
    imf_T_sim1(:, uu) = double(T_sim1_3);  % cell2mat将cell元胞数组转换为普通数组
    imf_T_sim2(:, uu) = double(T_sim2_3);
end

%% 重构出到真实的测试集与训练集
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);

%% 训练集和测试集划分%% 来自：公众号《淘个代码》
T_train_11 = res(1:num_train_s, 2);
T_train_11 = cell2mat(T_train_11)';

T_test_11 = res(num_train_s+1:end, 2);
T_test_11 = cell2mat(T_test_11)';

%% 各分量预测的结果相加
T_sim_a_11 = sum(imf_T_sim1, 2);
T_sim_b_11 = sum(imf_T_sim2, 2);
% 指标计算
disp('…………训练集误差指标…………')
[mae21, rmse21, mape21, error21] = calc_error(T_train_11, T_sim_a_11');
fprintf('\n')

% -------------------训练集预测效果对比图-------------------
figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置图形尺寸和背景色
plot(T_train_11, 'LineWidth', 1.5); % 绘制真实值，设置线宽
hold on;
plot(T_sim_a_11', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
title('CEEMDAN-VMD-CNN-biLSTM训练集预测效果对比', 'FontSize', 14);
xlabel('样本点', 'FontSize', 12);
ylabel('发电功率', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12); % 设置坐标轴字体大小

disp('…………测试集误差指标…………')
[mae22, rmse22, mape22, error22] = calc_error(T_test_11, T_sim_b_11');
fprintf('\n')

% -------------------测试集预测效果对比图-------------------
figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
plot(T_test_11, 'LineWidth', 1.5); % 绘制真实值，设置线宽
hold on;
plot(T_sim_b_11', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
title('CEEMDAN-VMD-CNN-biLSTM预测集预测效果对比', 'FontSize', 14);
xlabel('样本点', 'FontSize', 12);
ylabel('发电功率', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12); % 设置坐标轴字体大小

% -------------------误差曲线图-------------------
figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
plot(T_sim_b_11' - T_test_11, 'LineWidth', 1.5); % 绘制预测误差
title('CEEMDAN-VMD-CNN-biLSTM误差曲线图', 'FontSize', 14);
xlabel('样本点', 'FontSize', 12);
ylabel('发电功率误差', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12); % 设置坐标轴字体大小

%%
figure('Position', [200, 300, 600, 400], 'Color', 'w');
scatter(T_test_11, T_sim_b_11', 30, 'filled'); % 真实值 vs 预测值
hold on;
plot([min(T_test_11) max(T_test_11)], [min(T_test_11) max(T_test_11)], 'r', 'LineWidth', 1.5); % 参考线
xlabel('真实值', 'FontSize', 12);
ylabel('预测值', 'FontSize', 12);
title('真实值 vs 预测值散点图', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);
legend('预测数据点', 'y=x参考线', 'FontSize', 12);

%%
figure('Position', [200, 300, 900, 300], 'Color', 'w');
bar(T_sim_b_11' - T_test_11, 'FaceColor', [0.2 0.6 0.8]); % 误差柱状图
xlabel('样本点', 'FontSize', 12);
ylabel('预测误差', 'FontSize', 12);
title('预测误差的柱状图', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

%% 
figure('Position', [200, 300, 600, 400], 'Color', 'w');
histogram(T_sim_b_11' - T_test_11, 30, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k');
xlabel('预测误差', 'FontSize', 12);
ylabel('频数', 'FontSize', 12);
title('预测误差分布直方图', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

%%
figure('Position', [200, 300, 600, 400], 'Color', 'w');
boxplot(T_sim_b_11' - T_test_11, 'Labels', {'误差分布'});
ylabel('误差', 'FontSize', 12);
title('误差箱线图', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);
