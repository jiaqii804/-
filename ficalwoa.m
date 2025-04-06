function fitness = fical(x)
%%  从主函数中获取训练数据
    Lp_train = evalin('base', 'Lp_train');
    t_train = evalin('base', 't_train');
    ps_output = evalin('base', 'ps_output');
    T_train = evalin('base', 'T_train');
    f_ = evalin('base', 'f_');
    
    best_hd  = round(x(1, 2)); % 最佳隐藏层节点数
    best_lr= x(1, 1);% 最佳初始学习率
    best_l2 = x(1, 3);% 最佳L2正则化系数

    %%  建立模型
    lgraph = layerGraph();                                                 % 建立空白网络结构
    
    tempLayers = [
        sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
        sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
    lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
    
    tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
    lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
     
    tempLayers = [
        reluLayer("Name", "relu_1")                                        % 激活层
        convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
        reluLayer("Name", "relu_2")];                                      % 激活层
    lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
    
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
        bilstmLayer(best_hd, "Name", "lstm", "OutputMode", "last")                 % bilstm层
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

    %%  参数设置
    options = trainingOptions('adam', ...      % Adam 梯度下降算法
        'MaxEpochs', 500, ...                 % 最大迭代次数
        'InitialLearnRate', best_lr, ...          % 初始学习率为0.01
        'LearnRateSchedule', 'piecewise', ...  % 学习率下降
        'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.5
        'LearnRateDropPeriod', 400, ...        % 经过700次训练后 学习率为 0.01 * 0.1
        'L2Regularization',best_l2,...
        'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
        'Verbose', false);

    %%  训练模型
    net = trainNetwork(Lp_train, t_train, lgraph, options);

    %%  模型预测
    t_sim1 = predict(net, Lp_train);

    %%  数据反归一化
    T_sim1 = mapminmax('reverse', t_sim1', ps_output);
    T_sim1=double(T_sim1);

    %%  计算适应度
    fitness = sqrt(sum((T_sim1 - T_train).^2)./length(T_sim1));

end