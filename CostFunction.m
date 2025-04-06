
function valError = CostFunction(optVars)
%% ��ֵ
L2Regularization =abs(optVars(1)); % ���򻯲���
InitialLearnRate=abs(optVars(2)); % ��ʼѧϰ��
NumOfUnits = abs(round(optVars(3))); % 
%%  ��ȡ����
vp_train = evalin('base', 'vp_train');
vt_train = evalin('base', 'vt_train');

%%  ����������������
inputSize    = size(vp_train{1}, 1);
numResponses = size(vt_train{1}, 1);

%%  ��������ṹ
opt.layers = [ ...
    sequenceInputLayer(inputSize)       % �����

    bilstmLayer(NumOfUnits)             % bilstm��

    selfAttentionLayer(1,2)             % ����һ����ͷ��2�����Ͳ�ѯͨ������ע������  

    reluLayer                           % Relu�����

    fullyConnectedLayer(numResponses)   % �ع��

    regressionLayer];

%%  �����������
opt.options = trainingOptions('adam', ...             % �Ż��㷨Adam
    'MaxEpochs', 100, ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', InitialLearnRate, ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', 80, ...                   % ѵ��85�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', L2Regularization, ...         % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 1, ...                                 % ����/�ر��Ż�����
    'Plots', 'none');                                 % ����������

%%  ѵ������
net = trainNetwork(vp_train, vt_train, opt.layers, opt.options);

%%  �õ�����Ԥ��ֵ
t_sim1 = predict(net, vp_train); 

%%  �������
valError = sqrt(double(mse(vt_train, t_sim1)));

end