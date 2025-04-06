clc;
clear;
close all;

% 读取数据
X = xlsread('风电场预测.xlsx');
X = X(5665:8640, :);  % 选取3月份数据，最后一列为预测值

% 设置参数
n_in = 5;  % 输入前5个时刻的数据
n_out = 1; % 单步预测
or_dim = size(X, 2);  % 特征数据维度
num_samples = length(X) - n_in; % 样本个数
scroll_window = 1;  % 滑动窗口

% 数据预处理
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);

% 划分训练集和测试集
num_size = 0.8;                              
num_train_s = round(num_size * num_samples); 

% 方便归一化
P_train = res(1:num_train_s, 1);
P_train = reshape(cell2mat(P_train)', n_in * or_dim, num_train_s);
T_train = res(1:num_train_s, 2);
T_train = cell2mat(T_train)';

P_test = res(num_train_s + 1:end, 1);
P_test = reshape(cell2mat(P_test)', n_in * or_dim, num_samples - num_train_s);
T_test = res(num_train_s + 1:end, 2);
T_test = cell2mat(T_test)';

% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax(T_test, ps_output);

 %%  数据平铺 %% 来自：公众号《淘个代码》

    for i = 1:size(P_train,2)
        Lp_train{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
    end

    for i = 1:size(p_test,2)
        Lp_test{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
    end
% 参数设置
targetD = t_train;
targetD_test = t_test;
numFeatures = size(p_train, 1);

% 数据重整
vp_train = reshape(p_train, n_in, or_dim, num_train_s);
vp_test = reshape(p_test, n_in, or_dim, num_samples - num_train_s);

% SSA参数设置
pop = 1; % 麻雀数量
Max_iteration = 1; % 最大迭代次数
dim = 3; % 优化LSTM的3个参数
lb = [40, 40, 0.001]; % 下边界
ub = [200, 200, 0.03]; % 上边界
% %归一化后的训练和测试集划分
xTrain=P_train;
xTest=P_test;
yTrain=T_train;
yTest=T_test;
numFeatures = size(xTrain,1);
numResponses = 1;


fobj = @(x) fun(x, numFeatures, numResponses, xTrain, yTrain, xTest, yTest);
[Best_pos, Best_score, curve] = SSA(pop, Max_iteration, lb, ub, dim, fobj); 

best_hd = round(Best_pos(2)); 
best_lr = round(Best_pos(1));
best_l2 = round(Best_pos(3));

% 网络结构保持不变
lgraph = layerGraph();
tempLayers = [
    sequenceInputLayer([numFeatures,1,1], "Name", "sequence")
    sequenceFoldingLayer("Name", "seqfold")];
lgraph = addLayers(lgraph, tempLayers);

tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name", "relu_1")
    convolution2dLayer([3, 1], 64, "Name", "conv_2")
    reluLayer("Name", "relu_2")];
lgraph = addLayers(lgraph, tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")
    fullyConnectedLayer(16, "Name", "fc_2")
    reluLayer("Name", "relu_3")
    fullyConnectedLayer(64, "Name", "fc_3")
    sigmoidLayer("Name", "sigmoid")];
lgraph = addLayers(lgraph, tempLayers);

tempLayers = multiplicationLayer(2, "Name", "multiplication");
lgraph = addLayers(lgraph, tempLayers);

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")
    flattenLayer("Name", "flatten")
    bilstmLayer(6, "Name", "lstm", "OutputMode", "last")
    fullyConnectedLayer(1, "Name", "fc")
    regressionLayer("Name", "regressionoutput")];
lgraph = addLayers(lgraph, tempLayers);

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
lgraph = connectLayers(lgraph, "conv_1", "relu_1");
lgraph = connectLayers(lgraph, "conv_1", "gapool");
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");

% 设置训练选项
options0 = trainingOptions('adam', ...
    'MaxEpochs', round(Best_pos(2)), ...
    'ExecutionEnvironment', 'cpu', ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', Best_pos(3), ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', round(0.8 * Best_pos(2)), ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', 0.0001, ...
    'Verbose', 0);

% 训练网络
net = trainNetwork(Lp_train, targetD', lgraph, options0);

% 预测
t_sim1 = predict(net, Lp_train);
t_sim2 = predict(net, Lp_test);

% 反归一化
T_sim_a_9 = mapminmax('reverse', t_sim1, ps_output);
T_sim_b_9 = mapminmax('reverse', t_sim2, ps_output);

%% 重构出到真实的测试集与训练集
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);
% 训练集和测试集划分%% 来自：公众号《淘个代码》
%% 以下几行代码是为了方便归一化，一般不需要更改！
T_train_9 = res(1: num_train_s,2);
T_train_9 = cell2mat(T_train_9)';

T_test_9 = res(num_train_s+1: end,2);
T_test_9 = cell2mat(T_test_9)';




% 指标计算
disp('…………训练集误差指标…………')
[mae17,rmse17,mape17,error17]=calc_error(T_train_9,T_sim_a_9');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train_9);
hold on
plot(T_sim_a_9')
legend('真实值','预测值')
title('SSA-CNN-biLSTM训练集预测效果对比')
xlabel('样本点')
ylabel('发电功率')

disp('…………测试集误差指标…………')
[mae18,rmse18,mape18,error18]=calc_error(T_test_9,T_sim_b_9');
fprintf('\n')


figure('Position',[200,300,600,200])
plot(T_test_9);
hold on
plot(T_sim_b_9')
legend('真实值','预测值')
title('CEEMDAN-VMD-SSA-CNN-biLSTM预测集预测效果对比')
xlabel('样本点')
ylabel('发电功率')

figure('Position',[200,300,600,200])
plot(T_sim_b_9'-T_test_9)
title('CEEMDAN-VMD-SSA-CNN-biLSTM-误差曲线图')
xlabel('样本点')
ylabel('发电功率')
