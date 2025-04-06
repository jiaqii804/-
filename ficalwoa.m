function fitness = fical(x)
%%  ���������л�ȡѵ������
    Lp_train = evalin('base', 'Lp_train');
    t_train = evalin('base', 't_train');
    ps_output = evalin('base', 'ps_output');
    T_train = evalin('base', 'T_train');
    f_ = evalin('base', 'f_');
    
    best_hd  = round(x(1, 2)); % ������ز�ڵ���
    best_lr= x(1, 1);% ��ѳ�ʼѧϰ��
    best_l2 = x(1, 3);% ���L2����ϵ��

    %%  ����ģ��
    lgraph = layerGraph();                                                 % �����հ�����ṹ
    
    tempLayers = [
        sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % ��������㣬�������ݽṹΪ[f_, 1, 1]
        sequenceFoldingLayer("Name", "seqfold")];                          % ���������۵���
    lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��
    
    tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % ����� �����[3, 1] ����[1, 1] ͨ���� 32
    lgraph = addLayers(lgraph,tempLayers);                                 % ����������ṹ����հ׽ṹ��
     
    tempLayers = [
        reluLayer("Name", "relu_1")                                        % �����
        convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % ����� �����[3, 1] ����[1, 1] ͨ���� 64
        reluLayer("Name", "relu_2")];                                      % �����
    lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��
    
    tempLayers = [
        globalAveragePooling2dLayer("Name", "gapool")                      % ȫ��ƽ���ػ���
        fullyConnectedLayer(16, "Name", "fc_2")                            % SEע�������ƣ�ͨ������1 / 4
        reluLayer("Name", "relu_3")                                        % �����
        fullyConnectedLayer(64, "Name", "fc_3")                            % SEע�������ƣ���Ŀ��ͨ������ͬ
        sigmoidLayer("Name", "sigmoid")];                                  % �����
    lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��
    
    tempLayers = multiplicationLayer(2, "Name", "multiplication");         % ��˵�ע����
    lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��
    
    tempLayers = [
        sequenceUnfoldingLayer("Name", "sequnfold")                        % �������з��۵���
        flattenLayer("Name", "flatten")                                    % ������ƽ��
        bilstmLayer(best_hd, "Name", "lstm", "OutputMode", "last")                 % bilstm��
        fullyConnectedLayer(1, "Name", "fc")                               % ȫ���Ӳ�
        regressionLayer("Name", "regressionoutput")];                      % �ع��
    lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��
    
    lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % �۵������ ���� ���������;
    lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                           % �۵������ ���� ���۵�������  
    lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % �������� ���� �����
    lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % �������� ���� ȫ��ƽ���ػ�
    lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % �������� ���� ��˲�
    lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % ȫ������� ���� ��˲�
    lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % ������

    %%  ��������
    options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
        'MaxEpochs', 500, ...                 % ����������
        'InitialLearnRate', best_lr, ...          % ��ʼѧϰ��Ϊ0.01
        'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
        'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.5
        'LearnRateDropPeriod', 400, ...        % ����700��ѵ���� ѧϰ��Ϊ 0.01 * 0.1
        'L2Regularization',best_l2,...
        'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
        'Verbose', false);

    %%  ѵ��ģ��
    net = trainNetwork(Lp_train, t_train, lgraph, options);

    %%  ģ��Ԥ��
    t_sim1 = predict(net, Lp_train);

    %%  ���ݷ���һ��
    T_sim1 = mapminmax('reverse', t_sim1', ps_output);
    T_sim1=double(T_sim1);

    %%  ������Ӧ��
    fitness = sqrt(sum((T_sim1 - T_train).^2)./length(T_sim1));

end