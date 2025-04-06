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
    n_out = 1 ; % 单步预测
    or_dim = size(X,2) ;       % 记录特征数据维度
    num_samples = length(X_imf)- n_in;
    scroll_window = 1;
    [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);

    num_size = 0.8;                             
    num_train_s = round(num_size * num_samples);

    P_train = res(1: num_train_s,1);
    P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
    T_train = res(1: num_train_s,2);
    T_train = cell2mat(T_train)';

    P_test = res(num_train_s+1: end,1);
    P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
    T_test = res(num_train_s+1: end,2);
    T_test = cell2mat(T_test)';

    % 数据归一化
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);

    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test = mapminmax('apply', T_test, ps_output);

    vp_train = reshape(p_train,n_in,or_dim,num_train_s);
    vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);

    vt_train = t_train;
    vt_test = t_test;

    %% SSA优化
    pop=3;
    Max_iteration=6;
    dim=3;
    lb = [40,40,0.001];
    ub = [200,200,0.03];

    xTrain=P_train;
    xTest=P_test;
    yTrain=T_train;
    yTest=T_test;
    numFeatures = size(xTrain,1);
    numResponses = 1;
    fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);
    [Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj);

    best_hd  = round(Best_pos(2));
    best_lr= round(Best_pos(1));
    best_l2 = round(Best_pos(3));

    %% 收敛曲线
    figure;
    plot(curve,'r','LineWidth',2);
    xlabel('迭代次数');
    ylabel('适应度值');
    title('SSA优化收敛曲线');
    legend('SSA优化过程');

    %% 训练网络
    options0 = trainingOptions('adam', ...
        'MaxEpochs',round(Best_pos(2)), ...
        'ExecutionEnvironment' ,'cpu',...
        'GradientThreshold',1, ...
        'InitialLearnRate',Best_pos(3), ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
        'LearnRateDropFactor',0.1, ...
        'L2Regularization',0.0001,...
        'Verbose',0);

    net = trainNetwork(Lp_train,targetD',lgraph,options0);
    t_sim1 = predict(net, Lp_train);
    t_sim2 = predict(net, Lp_test);

    T_sim1_4 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2_4 = mapminmax('reverse', t_sim2, ps_output);

    imf_T_sim1(:,uu) = double(T_sim1_4);
    imf_T_sim2(:,uu) = double(T_sim2_4);
end

%% 误差分析
T_sim_a_4 = sum(imf_T_sim1,2);
T_sim_b_4 = sum(imf_T_sim2,2);

[mae7,rmse7,mape7,error7]=calc_error(T_train_4,T_sim_a_4');
figure;
plot(error7,'b','LineWidth',1.5);
xlabel('样本点');
ylabel('误差值');
title('训练集误差曲线');
legend('误差');

[mae8,rmse8,mape8,error8]=calc_error(T_test_4,T_sim_b_4');
figure;
plot(error8,'b','LineWidth',1.5);
xlabel('样本点');
ylabel('误差值');
title('测试集误差曲线');
legend('误差');
