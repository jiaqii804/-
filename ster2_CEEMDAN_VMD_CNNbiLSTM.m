%% 此程序为多特征输入，单步预测CEEMDAN-VMD-SSA-CNNbiLSTM-ATTENTION
clc;
clear
close all
X = xlsread('风电场预测.xlsx');
X = X(5665:8640,:);  %选取3月份数据

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
    or_dim = size(X,2) ;       % 记录特征数据维度
    num_samples = length(X_imf)- n_in; % 样本个数
    scroll_window = 1;  %如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
    [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);



    num_size = 0.8;                             
    num_train_s = round(num_size * num_samples); 

  
    P_train = res(1: num_train_s,1)
    P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
    T_train = res(1: num_train_s,2);
    T_train = cell2mat(T_train)';

    P_test = res(num_train_s+1: end,1);
    P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
    T_test = res(num_train_s+1: end,2);
    T_test = cell2mat(T_test)';


    %  数据归一化
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);

    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test = mapminmax('apply', T_test, ps_output);

    vp_train = reshape(p_train,n_in,or_dim,num_train_s);
    vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);

    vt_train = t_train;
    vt_test = t_test;

    f_ = [size(vp_train,1) size(vp_train,2)];
    outdim = n_out;

    %%  数据平铺 

    for i = 1:size(P_train,2)
        Lp_train{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
    end

    for i = 1:size(p_test,2)
        Lp_test{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
    end


%% 参数设置
    targetD =  t_train;
    targetD_test  =  t_test;

    numFeatures = size(p_train,1);
%% 小麻雀SSA
pop=3; % 麻雀数量
Max_iteration=6; % 最大迭代次数
dim=3; % 优化lstm的3个参数
lb = [40,40,0.001];%下边界
ub = [200,200,0.03];%上边界

% %归一化后的训练和测试集划分
xTrain=P_train;
xTest=P_test;
yTrain=T_train;
yTest=T_test;
numFeatures = size(xTrain,1);
numResponses = 1;
fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);%xTrain:归一化后的训练和测试集划分
 % [Best_score,Best_pos,curve]=WOA(SearchAgents_no,Max_iteration,lb ,ub,dim,finess);
[Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化SSA
% 
best_hd  = round(Best_pos(2)); % 最佳隐藏层节点数
best_lr= round(Best_pos(1));% 最佳初始学习率
best_l2 = round(Best_pos(3));% 最佳L2正则化系数



    %% 预测模型
    lgraph = layerGraph();                                                 % 建立空白网络结构
 % 输入特征
tempLayers = [
    sequenceInputLayer([numFeatures,1,1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
  % CNN特征提取
tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中

tempLayers = [
    reluLayer("Name", "relu_1")                                        % 激活层
    convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
    reluLayer("Name", "relu_2")];                                      % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 池化层
tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
    reluLayer("Name", "relu_3")                                        % 激活层
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
    sigmoidLayer("Name", "sigmoid")];                                  % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
     bilstmLayer(6, "Name", "lstm", "OutputMode", "last")                 % bilstm层//（比LSTM多）不同点

    fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
    regressionLayer("Name", "regressionoutput")];                      % 回归层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % 折叠层输出 连接 反折叠层输入  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出


    %% 参数设置SSA
    % 指定训练选项SSA
    options0 = trainingOptions('adam', ...
        'MaxEpochs',round(Best_pos(2)), ...%最大迭代次数
        'ExecutionEnvironment' ,'cpu',...
        'GradientThreshold',1, ...
        'InitialLearnRate',Best_pos(3), ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
        'LearnRateDropFactor',0.1, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
        'L2Regularization',0.0001,...
        'Verbose',0);
    % 优化算法：使用Adam优化算法（'adam'）。
    % 最大迭代次数：MaxEpochs 设置为 Best_pos(2) 的四舍五入值，即训练过程中的最大迭代次数由 Best_pos(2) 决定。
    % 执行环境：ExecutionEnvironment 设置为 'cpu'，表示训练过程将在CPU上执行，不使用GPU加速。
    % 梯度阈值：GradientThreshold 设置为1，用于裁剪梯度爆炸问题。
    % 初始学习率：InitialLearnRate 设置为 Best_pos(3)，即初始学习率由 Best_pos(3) 决定。
    % 学习率调度策略：LearnRateSchedule 设置为 'piecewise'，表示学习率将按照分段函数的方式调整。
    % 学习率下降周期：LearnRateDropPeriod 设置为 round(0.8*Best_pos(2))，即学习率下降的周期是 Best_pos(2) 的80%的四舍五入值。
    % 学习率下降因子：LearnRateDropFactor 设置为 0.2，表示在每个下降周期后，学习率将乘以0.2来降低。
    % L2正则化：L2Regularization 设置为 0.0001，用于防止过拟合。
    % 详细输出：Verbose 设置为 0，表示在训练过程中不输出详细的训练信息

    %% start training
    %  训练
    net = trainNetwork(Lp_train,targetD',lgraph,options0);
    % net = trainNetwork(trainD,targetD',lgraph0,options0);
    %analyzeNetwork(net);% 查看网络结构
    %  预测
    t_sim1 = predict(net, Lp_train);
    t_sim2 = predict(net, Lp_test);

    %  数据反归一化
    T_sim1_4 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2_4 = mapminmax('reverse', t_sim2, ps_output);

    %  数据格式转换
    imf_T_sim1(:,uu) = double(T_sim1_4);% cell2mat将cell元胞数组转换为普通数组
    imf_T_sim2(:,uu) = double(T_sim2_4);
end

%% 重构出到真实的测试集与训练集
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);

%% 不动
T_train_4 = res(1: num_train_s,2);
T_train_4 = cell2mat(T_train_4)';

T_test_4 = res(num_train_s+1: end,2);
T_test_4 = cell2mat(T_test_4)';

%% 各分量预测的结果相加
T_sim_a_4 = sum(imf_T_sim1,2);
T_sim_b_4 = sum(imf_T_sim2,2);


% 指标计算
disp('…………训练集误差指标…………')
[mae7,rmse7,mape7,error7]=calc_error(T_train_4,T_sim_a_4');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train_4);
hold on
plot(T_sim_a_4')
legend('真实值','预测值')
title('CEEMDAN-VMD-SSA-CNN-biLSTM训练集预测效果对比')
xlabel('样本点')
ylabel('发电功率')

disp('…………测试集误差指标…………')
[mae8,rmse8,mape8,error8]=calc_error(T_test_4,T_sim_b_4');
fprintf('\n')


figure('Position',[200,300,600,200])
plot(T_test_4);
hold on
plot(T_sim_b_4')
legend('真实值','预测值')
title('CEEMDAN-VMD-SSA-CNN-biLSTM预测集预测效果对比')
xlabel('样本点')
ylabel('发电功率')

figure('Position',[200,300,600,200])
plot(T_sim_b_4'-T_test_4)
title('CEEMDAN-VMD-SSA-CNN-biLSTM-误差曲线图')
xlabel('样本点')
ylabel('发电功率')

%%
disp('…………训练集误差指标…………')
[mae7,rmse7,mape7,error7]=calc_error(T_train_4,T_sim_a_4');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train_4);
hold on
plot(T_sim_a_4')
legend('真实值','预测值')
title('CEEMDAN-VMD-SSA-CNN-biLSTM训练集预测效果对比')
xlabel('样本点')
ylabel('发电功率')

disp('…………测试集误差指标…………')
[mae8,rmse8,mape8,error8]=calc_error(T_test_4,T_sim_b_4');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_test_4);
hold on
plot(T_sim_b_4')
legend('真实值','预测值')
title('CEEMDAN-VMD-SSA-CNN-biLSTM预测集预测效果对比')
xlabel('样本点')
ylabel('发电功率')

figure('Position',[200,300,600,200])
plot(T_sim_b_4'-T_test_4)
title('CEEMDAN-VMD-SSA-CNN-biLSTM 误差曲线图')
xlabel('样本点')
ylabel('误差')

%% **新增散点图：真实值 vs. 预测值**
figure('Position',[200,300,600,400])
scatter(T_test_4, T_sim_b_4', 'filled');
xlabel('真实值')
ylabel('预测值')
title('真实值 vs. 预测值 (散点图)')
grid on
axis equal
hold on
plot(min(T_test_4):max(T_test_4), min(T_test_4):max(T_test_4), 'r--') % 参考线
legend('预测点', '理想预测')

%% **新增柱状图：误差指标比较**
figure('Position',[200,300,600,400])
bar([mae7, rmse7, mape7; mae8, rmse8, mape8])
set(gca, 'XTickLabel', {'训练集', '测试集'})
legend('MAE', 'RMSE', 'MAPE')
title('训练集 vs. 测试集 误差对比（柱状图）')
ylabel('误差值')

%% **新增箱线图：误差分布**
figure('Position',[200,300,600,400])
boxplot(T_sim_b_4' - T_test_4)
title('误差分布 (箱线图)')
ylabel('误差值')

%% **新增直方图：误差分布**
figure('Position',[200,300,600,400])
histogram(T_sim_b_4' - T_test_4, 30)
title('误差分布 (直方图)')
xlabel('误差值')
ylabel('频数')
grid on
