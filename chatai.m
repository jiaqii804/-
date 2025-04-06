% %% VMD-SSA-CNN-BI-AT（1516）
% clc;
% clear
% close all
% X = xlsread('风电场预测.xlsx');
% X = X(5665:8640,:);  %选取3月份数据，最后一列要是预测值哦
% load vmd1_data.mat
% IMF = u';
% disp('…………………………………………………………………………………………………………………………')
% disp('VMD-SSA-CNNbiLSTM预测')
% disp('…………………………………………………………………………………………………………………………')
% 
% %% 对每个分量建模
% for uu=1:size(IMF,2)
%     X_imf=[X(:,1:end-1),IMF(:,uu)];
% 
% 
%     n_in = 5;  % 输入前5个时刻的数据
%     n_out = 1 ; % 此程序为单步预测，因此请将n_out设置为1
%     or_dim = size(X,2) ;       % 记录特征数据维度
%     num_samples = length(X_imf)- n_in; % 样本个数
%     scroll_window = 1;  %如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
%     [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);
% 
% 
%     %% 训练集和测试集划分%% 来自：公众号《淘个代码》
% 
%     num_size = 0.8;                              % 训练集占数据集比例  %% 来自：公众号《淘个代码》
%     num_train_s = round(num_size * num_samples); % 训练集样本个数  %% 来自：公众号《淘个代码》
% 
%     %% 以下几行代码是为了方便归一化，一般不需要更改！
%     P_train = res(1: num_train_s,1)
%     P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
%     T_train = res(1: num_train_s,2);
%     T_train = cell2mat(T_train)';
% 
%     P_test = res(num_train_s+1: end,1);
%     P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
%     T_test = res(num_train_s+1: end,2);
%     T_test = cell2mat(T_test)';
% 
% 
%     %  数据归一化
%     [p_train, ps_input] = mapminmax(P_train, 0, 1);
%     p_test = mapminmax('apply', P_test, ps_input);
% 
%     [t_train, ps_output] = mapminmax(T_train, 0, 1);
%     t_test = mapminmax('apply', T_test, ps_output);
% 
%     vp_train = reshape(p_train,n_in,or_dim,num_train_s);
%     vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);
% 
%     vt_train = t_train;
%     vt_test = t_test;
% 
%     f_ = [size(vp_train,1) size(vp_train,2)];
%     outdim = n_out;
% 
%     %%  数据平铺 %% 来自：公众号《淘个代码》
% 
%     for i = 1:size(P_train,2)
%         Lp_train{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
%     end
% 
%     for i = 1:size(p_test,2)
%         Lp_test{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
%     end
% 
% 
% %% 参数设置
%     targetD =  t_train;
%     targetD_test  =  t_test;
% 
%     numFeatures = size(p_train,1);
% %% 小麻雀SSA
% pop=1; % 麻雀数量
% Max_iteration=1; % 最大迭代次数
% dim=3; % 优化lstm的3个参数
% lb = [40,40,0.001];%下边界
% ub = [200,200,0.03];%上边界
% 
% % %归一化后的训练和测试集划分
% xTrain=P_train;
% xTest=P_test;
% yTrain=T_train;
% yTest=T_test;
% numFeatures = size(xTrain,1);
% numResponses = 1;
% fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);%xTrain:归一化后的训练和测试集划分
%  % [Best_score,Best_pos,curve]=WOA(SearchAgents_no,Max_iteration,lb ,ub,dim,finess);
% [Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化SSA
% % 
% best_hd  = round(Best_pos(2)); % 最佳隐藏层节点数
% best_lr= round(Best_pos(1));% 最佳初始学习率
% best_l2 = round(Best_pos(3));% 最佳L2正则化系数
% 
% 
% 
%     %% ATTENTION
%     lgraph = layerGraph();                                                 % 建立空白网络结构
%  % 输入特征
% tempLayers = [
%     sequenceInputLayer([numFeatures,1,1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
%     sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
%   % CNN特征提取
% tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
% lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     reluLayer("Name", "relu_1")                                        % 激活层
%     convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
%     reluLayer("Name", "relu_2")];                                      % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% % 池化层
% tempLayers = [
%     globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
%     fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
%     reluLayer("Name", "relu_3")                                        % 激活层
%     fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
%     sigmoidLayer("Name", "sigmoid")];                                  % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
%     flattenLayer("Name", "flatten")                                    % 网络铺平层
%      bilstmLayer(6, "Name", "lstm", "OutputMode", "last")                 % bilstm层//（比LSTM多）不同点
% 
%     fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
%     regressionLayer("Name", "regressionoutput")];                      % 回归层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
% lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
%                                                                        % 折叠层输出 连接 反折叠层输入  
% lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
% lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
% lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
% lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
% lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出
% 
% 
%     %% 参数设置SSA
%     % 指定训练选项SSA
%     options0 = trainingOptions('adam', ...
%         'MaxEpochs',round(Best_pos(2)), ...%最大迭代次数
%         'ExecutionEnvironment' ,'cpu',...
%         'GradientThreshold',1, ...
%         'InitialLearnRate',Best_pos(3), ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
%         'LearnRateDropFactor',0.1, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
%         'L2Regularization',0.0001,...
%         'Verbose',0);
%     % 优化算法：使用Adam优化算法（'adam'）。
%     % 最大迭代次数：MaxEpochs 设置为 Best_pos(2) 的四舍五入值，即训练过程中的最大迭代次数由 Best_pos(2) 决定。
%     % 执行环境：ExecutionEnvironment 设置为 'cpu'，表示训练过程将在CPU上执行，不使用GPU加速。
%     % 梯度阈值：GradientThreshold 设置为1，用于裁剪梯度爆炸问题。
%     % 初始学习率：InitialLearnRate 设置为 Best_pos(3)，即初始学习率由 Best_pos(3) 决定。
%     % 学习率调度策略：LearnRateSchedule 设置为 'piecewise'，表示学习率将按照分段函数的方式调整。
%     % 学习率下降周期：LearnRateDropPeriod 设置为 round(0.8*Best_pos(2))，即学习率下降的周期是 Best_pos(2) 的80%的四舍五入值。
%     % 学习率下降因子：LearnRateDropFactor 设置为 0.2，表示在每个下降周期后，学习率将乘以0.2来降低。
%     % L2正则化：L2Regularization 设置为 0.0001，用于防止过拟合。
%     % 详细输出：Verbose 设置为 0，表示在训练过程中不输出详细的训练信息
% 
%     %% start training
%     %  训练
%     net = trainNetwork(Lp_train,targetD',lgraph,options0);
%     % net = trainNetwork(trainD,targetD',lgraph0,options0);
%     %analyzeNetwork(net);% 查看网络结构
%     %  预测
%     t_sim1 = predict(net, Lp_train);
%     t_sim2 = predict(net, Lp_test);
% 
%     %  数据反归一化
%     T_sim1_4 = mapminmax('reverse', t_sim1, ps_output);
%     T_sim2_4 = mapminmax('reverse', t_sim2, ps_output);
% 
%     %  数据格式转换
%     imf_T_sim1(:,uu) = double(T_sim1_4);% cell2mat将cell元胞数组转换为普通数组
%     imf_T_sim2(:,uu) = double(T_sim2_4);
% end
% 
% %% 重构出到真实的测试集与训练集
% [res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);
% % 训练集和测试集划分%% 来自：公众号《淘个代码》
% %% 以下几行代码是为了方便归一化，一般不需要更改！
% T_train_8 = res(1: num_train_s,2);
% T_train_8 = cell2mat(T_train_8)';
% 
% T_test_8 = res(num_train_s+1: end,2);
% T_test_8 = cell2mat(T_test_8)';
% 
% %% 各分量预测的结果相加
% T_sim_a_8 = sum(imf_T_sim1,2);
% T_sim_b_8 = sum(imf_T_sim2,2);
% % 
% % % 
% % -------------------训练集误差指标计算-------------------
% disp('…………训练集误差指标…………')
% [mae15, rmse15, mape15, error15] = calc_error(T_train_8, T_sim_a_8');
% fprintf('\n')

% % -------------------训练集预测效果对比图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_train_8, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% hold on;
% plot(T_sim_a_8', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% title('CEEMDAN-VMD-SSA-CNN-biLSTM训练集预测效果对比', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% % -------------------测试集误差指标计算-------------------
% disp('…………测试集误差指标…………')
% [mae16, rmse16, mape16, error16] = calc_error(T_test_8, T_sim_b_8');
% fprintf('\n')
% 
% % -------------------测试集预测效果对比图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_test_8, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% hold on;
% plot(T_sim_b_8', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% title('VMD-SSA-CNN-biLSTM测试集预测效果对比', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% % -------------------测试集误差曲线图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_sim_b_8' - T_test_8, 'LineWidth', 1.5); % 绘制预测误差
% title('VMD-SSA-CNN-biLSTM误差曲线图', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率误差', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小

%%
% ------------------- Training Set Prediction Performance Comparison -------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% plot(T_train_8, 'LineWidth', 1.5); % Plot real values, set line width
% hold on;
% plot(T_sim_a_8', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
% legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
% title('CEEMDAN-VMD-SSA-CNN-biLSTM Training Set Prediction Performance Comparison', 'FontSize', 14);
% xlabel('Sample Points', 'FontSize', 12);
% ylabel('Power Output', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % Set axis font size
% 
% % ------------------- Test Set Error Metrics -------------------
% disp('………… Test Set Error Metrics …………')
% [mae16, rmse16, mape16, error16] = calc_error(T_test_8, T_sim_b_8');
% fprintf('\n')
% 
% % ------------------- Test Set Prediction Performance Comparison -------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% plot(T_test_8, 'LineWidth', 1.5); % Plot real values, set line width
% hold on;
% plot(T_sim_b_8', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
% legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
% title('VMD-SSA-CNN-biLSTM Test Set Prediction Performance Comparison', 'FontSize', 14);
% xlabel('Sample Points', 'FontSize', 12);
% ylabel('Power Output', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % Set axis font size
% 
% % ------------------- Test Set Error Curve -------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% plot(T_sim_b_8' - T_test_8, 'LineWidth', 1.5); % Plot prediction errors
% title('VMD-SSA-CNN-biLSTM Error Curve', 'FontSize', 14);
% xlabel('Sample Points', 'FontSize', 12);
% ylabel('Power Output Error', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % Set axis font size


%% CEEMDAN+预测（1314）
% clc;
% clear
% close all
% X = xlsread('风电场预测.xlsx');
% X = X(5665:8640,:);  %选取3月份数据，最后一列要是预测值哦
% load ceemdan_results.mat
% IMF = modes';
% disp('…………………………………………………………………………………………………………………………')
% disp('CEEMDAN-SSA-CNNbiLSTM预测')
% disp('…………………………………………………………………………………………………………………………')
% 
% %% 对每个分量建模
% for uu=1:size(IMF,2)
%     X_imf=[X(:,1:end-1),IMF(:,uu)];
% 
% 
%     n_in = 5;  % 输入前5个时刻的数据
%     n_out = 1 ; % 此程序为单步预测，因此请将n_out设置为1
%     or_dim = size(X,2) ;       % 记录特征数据维度
%     num_samples = length(X_imf)- n_in; % 样本个数
%     scroll_window = 1;  %如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
%     [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);
% 
% 
%     %% 训练集和测试集划分%% 来自：公众号《淘个代码》
% 
%     num_size = 0.8;                              % 训练集占数据集比例  %% 来自：公众号《淘个代码》
%     num_train_s = round(num_size * num_samples); % 训练集样本个数  %% 来自：公众号《淘个代码》
% 
%     %% 以下几行代码是为了方便归一化，一般不需要更改！
%     P_train = res(1: num_train_s,1)
%     P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
%     T_train = res(1: num_train_s,2);
%     T_train = cell2mat(T_train)';
% 
%     P_test = res(num_train_s+1: end,1);
%     P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
%     T_test = res(num_train_s+1: end,2);
%     T_test = cell2mat(T_test)';
% 
% 
%     %  数据归一化
%     [p_train, ps_input] = mapminmax(P_train, 0, 1);
%     p_test = mapminmax('apply', P_test, ps_input);
% 
%     [t_train, ps_output] = mapminmax(T_train, 0, 1);
%     t_test = mapminmax('apply', T_test, ps_output);
% 
%     vp_train = reshape(p_train,n_in,or_dim,num_train_s);
%     vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);
% 
%     vt_train = t_train;
%     vt_test = t_test;
% 
%     f_ = [size(vp_train,1) size(vp_train,2)];
%     outdim = n_out;
% 
%     %%  数据平铺 %% 来自：公众号《淘个代码》
% 
%     for i = 1:size(P_train,2)
%         Lp_train{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
%     end
% 
%     for i = 1:size(p_test,2)
%         Lp_test{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
%     end
% 
% 
% %% 参数设置
%     targetD =  t_train;
%     targetD_test  =  t_test;
% 
%     numFeatures = size(p_train,1);
% %% 小麻雀SSA
% pop=1; % 麻雀数量
% Max_iteration=1; % 最大迭代次数
% dim=3; % 优化lstm的3个参数
% lb = [40,40,0.001];%下边界
% ub = [200,200,0.03];%上边界
% 
% % %归一化后的训练和测试集划分
% xTrain=P_train;
% xTest=P_test;
% yTrain=T_train;
% yTest=T_test;
% numFeatures = size(xTrain,1);
% numResponses = 1;
% fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);%xTrain:归一化后的训练和测试集划分
%  % [Best_score,Best_pos,curve]=WOA(SearchAgents_no,Max_iteration,lb ,ub,dim,finess);
% [Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化SSA
% % 
% best_hd  = round(Best_pos(2)); % 最佳隐藏层节点数
% best_lr= round(Best_pos(1));% 最佳初始学习率
% best_l2 = round(Best_pos(3));% 最佳L2正则化系数
% 
% 
% 
%     %% ATTENTION
%     lgraph = layerGraph();                                                 % 建立空白网络结构
%  % 输入特征
% tempLayers = [
%     sequenceInputLayer([numFeatures,1,1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
%     sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
%   % CNN特征提取
% tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
% lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     reluLayer("Name", "relu_1")                                        % 激活层
%     convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
%     reluLayer("Name", "relu_2")];                                      % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% % 池化层
% tempLayers = [
%     globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
%     fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
%     reluLayer("Name", "relu_3")                                        % 激活层
%     fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
%     sigmoidLayer("Name", "sigmoid")];                                  % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
%     flattenLayer("Name", "flatten")                                    % 网络铺平层
%      bilstmLayer(6, "Name", "lstm", "OutputMode", "last")                 % bilstm层//（比LSTM多）不同点
% 
%     fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
%     regressionLayer("Name", "regressionoutput")];                      % 回归层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
% lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
%                                                                        % 折叠层输出 连接 反折叠层输入  
% lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
% lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
% lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
% lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
% lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出
% 
% 
%     %% 参数设置SSA
%     % 指定训练选项SSA
%     options0 = trainingOptions('adam', ...
%         'MaxEpochs',round(Best_pos(2)), ...%最大迭代次数
%         'ExecutionEnvironment' ,'cpu',...
%         'GradientThreshold',1, ...
%         'InitialLearnRate',Best_pos(3), ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
%         'LearnRateDropFactor',0.1, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
%         'L2Regularization',0.0001,...
%         'Verbose',0);
%     % 优化算法：使用Adam优化算法（'adam'）。
%     % 最大迭代次数：MaxEpochs 设置为 Best_pos(2) 的四舍五入值，即训练过程中的最大迭代次数由 Best_pos(2) 决定。
%     % 执行环境：ExecutionEnvironment 设置为 'cpu'，表示训练过程将在CPU上执行，不使用GPU加速。
%     % 梯度阈值：GradientThreshold 设置为1，用于裁剪梯度爆炸问题。
%     % 初始学习率：InitialLearnRate 设置为 Best_pos(3)，即初始学习率由 Best_pos(3) 决定。
%     % 学习率调度策略：LearnRateSchedule 设置为 'piecewise'，表示学习率将按照分段函数的方式调整。
%     % 学习率下降周期：LearnRateDropPeriod 设置为 round(0.8*Best_pos(2))，即学习率下降的周期是 Best_pos(2) 的80%的四舍五入值。
%     % 学习率下降因子：LearnRateDropFactor 设置为 0.2，表示在每个下降周期后，学习率将乘以0.2来降低。
%     % L2正则化：L2Regularization 设置为 0.0001，用于防止过拟合。
%     % 详细输出：Verbose 设置为 0，表示在训练过程中不输出详细的训练信息
% 
%     %% start training
%     %  训练
%     net = trainNetwork(Lp_train,targetD',lgraph,options0);
%     % net = trainNetwork(trainD,targetD',lgraph0,options0);
%     %analyzeNetwork(net);% 查看网络结构
%     %  预测
%     t_sim1 = predict(net, Lp_train);
%     t_sim2 = predict(net, Lp_test);
% 
%     %  数据反归一化
%     T_sim1_4 = mapminmax('reverse', t_sim1, ps_output);
%     T_sim2_4 = mapminmax('reverse', t_sim2, ps_output);
% 
%     %  数据格式转换
%     imf_T_sim1(:,uu) = double(T_sim1_4);% cell2mat将cell元胞数组转换为普通数组
%     imf_T_sim2(:,uu) = double(T_sim2_4);
% end
% 
% %% 重构出到真实的测试集与训练集
% [res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);
% % 训练集和测试集划分%% 来自：公众号《淘个代码》
% %% 以下几行代码是为了方便归一化，一般不需要更改！
% T_train_7 = res(1: num_train_s,2);
% T_train_7 = cell2mat(T_train_7)';
% 
% T_test_7 = res(num_train_s+1: end,2);
% T_test_7 = cell2mat(T_test_7)';
% 
% %% 各分量预测的结果相加
% T_sim_a_7 = sum(imf_T_sim1,2);
% T_sim_b_7 = sum(imf_T_sim2,2);
% 
% % % ------------------- Error Metrics Calculation -------------------
% disp('………… Training Set Error Metrics …………')
% [mae13, rmse13, mape13, error13] = calc_error(T_train_7, T_sim_a_7');
% fprintf('\n')
% 
% % ------------------- Training Set Prediction Performance Comparison -------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% plot(T_train_7, 'LineWidth', 1.5); % Plot real values, set line width
% hold on;
% plot(T_sim_a_7', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
% legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
% title('CEEMDAN-SSA-CNN-biLSTM Training Set Prediction Performance Comparison', 'FontSize', 14);
% xlabel('Sample Points', 'FontSize', 12);
% ylabel('Power Output', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % Set axis font size
% 
% disp('………… Test Set Error Metrics …………')
% [mae14, rmse14, mape14, error14] = calc_error(T_test_7, T_sim_b_7');
% fprintf('\n')
% 
% % ------------------- Test Set Prediction Performance Comparison -------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% plot(T_test_7, 'LineWidth', 1.5); % Plot real values, set line width
% hold on;
% plot(T_sim_b_7', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
% legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
% title('CEEMDAN-SSA-CNN-biLSTM Test Set Prediction Performance Comparison', 'FontSize', 14);
% xlabel('Sample Points', 'FontSize', 12);
% ylabel('Power Output', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % Set axis font size
% 
% % ------------------- Error Curve -------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% plot(T_sim_b_7' - T_test_7, 'LineWidth', 1.5); % Plot prediction errors
% title('CEEMDAN-SSA-CNN-biLSTM Error Curve', 'FontSize', 14);
% xlabel('Sample Points', 'FontSize', 12);
% ylabel('Power Output Error', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % Set axis font size
% 
% % % -------------------指标计算-------------------
% % disp('…………训练集误差指标…………')
% % [mae13, rmse13, mape13, error13] = calc_error(T_train_7, T_sim_a_7');
% % fprintf('\n')
% % 
% % % -------------------训练集预测效果对比图-------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 调整图形尺寸，设置底色为白色
% % plot(T_train_7, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% % hold on;
% % plot(T_sim_a_7', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% % legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% % title('CEEMDAN-SSA-CNN-biLSTM训练集预测效果对比', 'FontSize', 14);
% % xlabel('样本点', 'FontSize', 12);
% % ylabel('发电功率', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% % 
% % disp('…………测试集误差指标…………')
% % [mae14, rmse14, mape14, error14] = calc_error(T_test_7, T_sim_b_7');
% % fprintf('\n')
% % 
% % % -------------------测试集预测效果对比图-------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% % plot(T_test_7, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% % hold on;
% % plot(T_sim_b_7', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% % legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% % title('CEEMDAN-SSA-CNN-biLSTM测试集预测效果对比', 'FontSize', 14);
% % xlabel('样本点', 'FontSize', 12);
% % ylabel('发电功率', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% % 
% % % -------------------误差曲线图-------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% % plot(T_sim_b_7' - T_test_7, 'LineWidth', 1.5); % 绘制预测误差
% % title('CEEMDAN-SSA-CNN-biLSTM误差曲线图', 'FontSize', 14);
% % xlabel('样本点', 'FontSize', 12);
% % ylabel('发电功率误差', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% % 
% % 
% % % 指标计算
% % disp('…………训练集误差指标…………')
% % [mae13,rmse13,mape13,error13]=calc_error(T_train_7,T_sim_a_7');
% % fprintf('\n')
% % 
% % figure('Position',[200,300,600,200])
% % plot(T_train_7);
% % hold on
% % plot(T_sim_a_7')
% % legend('真实值','预测值')
% % title('CEEMDAN-SSA-CNN-biLSTM训练集预测效果对比')
% % xlabel('样本点')
% % ylabel('发电功率')
% % 
% % disp('…………测试集误差指标…………')
% % [mae14,rmse14,mape14,error14]=calc_error(T_test_7,T_sim_b_7');
% % fprintf('\n')
% % 
% % 
% % figure('Position',[200,300,600,200])
% % plot(T_test_7);
% % hold on
% % plot(T_sim_b_7')
% % legend('真实值','预测值')
% % title('CEEMDAN-SSA-CNN-biLSTM预测集预测效果对比')
% % xlabel('样本点')
% % ylabel('发电功率')
% % 
% % figure('Position',[200,300,600,200])
% % plot(T_sim_b_7'-T_test_7)
% % title('CEEMDAN-VMD-SSA-CNN-biLSTM-误差曲线图')
% % xlabel('样本点')
% % ylabel('发电功率')
%
%% EMD（1112)
% clc;
% clear;
% close all;
% 
% %加载数据
% X = xlsread('风电场预测.xlsx');
% X = X(5665:8640, :);  % 选取3月份数据，最后一列为预测值
% 
% % 使用EMD分解最后一列的预测值数据
% IMF = emd(X(:, end));  % 对预测值进行EMD分解
% IMF = IMF';            % 转置以便后续处理
% 
% disp('…………………………………………………………………………………………………………………………')
% disp('EMD-SSA-CNNbiLSTM预测')
% disp('…………………………………………………………………………………………………………………………')
% %% 对每个分量建模
% for uu=1:size(IMF,2)
%     X_imf=[X(:,1:end-1),IMF(:,uu)];
% 
% 
%     n_in = 5;  % 输入前5个时刻的数据
%     n_out = 1 ; % 此程序为单步预测，因此请将n_out设置为1
%     or_dim = size(X,2) ;       % 记录特征数据维度
%     num_samples = length(X_imf)- n_in; % 样本个数
%     scroll_window = 1;  %如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
%     [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);
% 
% 
%     %% 训练集和测试集划分%% 来自：公众号《淘个代码》
% 
%     num_size = 0.8;                              % 训练集占数据集比例  %% 来自：公众号《淘个代码》
%     num_train_s = round(num_size * num_samples); % 训练集样本个数  %% 来自：公众号《淘个代码》
% 
%     %% 以下几行代码是为了方便归一化，一般不需要更改！
%     P_train = res(1: num_train_s,1)
%     P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
%     T_train = res(1: num_train_s,2);
%     T_train = cell2mat(T_train)';
% 
%     P_test = res(num_train_s+1: end,1);
%     P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
%     T_test = res(num_train_s+1: end,2);
%     T_test = cell2mat(T_test)';
% 
% 
%     %  数据归一化
%     [p_train, ps_input] = mapminmax(P_train, 0, 1);
%     p_test = mapminmax('apply', P_test, ps_input);
% 
%     [t_train, ps_output] = mapminmax(T_train, 0, 1);
%     t_test = mapminmax('apply', T_test, ps_output);
% 
%     vp_train = reshape(p_train,n_in,or_dim,num_train_s);
%     vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);
% 
%     vt_train = t_train;
%     vt_test = t_test;
% 
%     f_ = [size(vp_train,1) size(vp_train,2)];
%     outdim = n_out;
% 
%     %%  数据平铺 %% 来自：公众号《淘个代码》
% 
%     for i = 1:size(P_train,2)
%         Lp_train{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
%     end
% 
%     for i = 1:size(p_test,2)
%         Lp_test{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
%     end
% 
% 
% %% 参数设置
%     targetD =  t_train;
%     targetD_test  =  t_test;
% 
%     numFeatures = size(p_train,1);
% %% 小麻雀SSA
% pop=1; % 麻雀数量
% Max_iteration=1; % 最大迭代次数
% dim=3; % 优化lstm的3个参数
% lb = [40,40,0.001];%下边界
% ub = [200,200,0.03];%上边界
% 
% % %归一化后的训练和测试集划分
% xTrain=P_train;
% xTest=P_test;
% yTrain=T_train;
% yTest=T_test;
% numFeatures = size(xTrain,1);
% numResponses = 1;
% fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);%xTrain:归一化后的训练和测试集划分
%  % [Best_score,Best_pos,curve]=WOA(SearchAgents_no,Max_iteration,lb ,ub,dim,finess);
% [Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化SSA
% % 
% best_hd  = round(Best_pos(2)); % 最佳隐藏层节点数
% best_lr= round(Best_pos(1));% 最佳初始学习率
% best_l2 = round(Best_pos(3));% 最佳L2正则化系数
% 
% 
% 
%     %% ATTENTION
%     lgraph = layerGraph();                                                 % 建立空白网络结构
%  % 输入特征
% tempLayers = [
%     sequenceInputLayer([numFeatures,1,1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
%     sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
%   % CNN特征提取
% tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
% lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     reluLayer("Name", "relu_1")                                        % 激活层
%     convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
%     reluLayer("Name", "relu_2")];                                      % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% % 池化层
% tempLayers = [
%     globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
%     fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
%     reluLayer("Name", "relu_3")                                        % 激活层
%     fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
%     sigmoidLayer("Name", "sigmoid")];                                  % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
%     flattenLayer("Name", "flatten")                                    % 网络铺平层
%      bilstmLayer(6, "Name", "lstm", "OutputMode", "last")                 % bilstm层//（比LSTM多）不同点
% 
%     fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
%     regressionLayer("Name", "regressionoutput")];                      % 回归层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
% lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
%                                                                        % 折叠层输出 连接 反折叠层输入  
% lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
% lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
% lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
% lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
% lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出
% 
% 
%     %% 参数设置SSA
%     % 指定训练选项SSA
%     options0 = trainingOptions('adam', ...
%         'MaxEpochs',round(Best_pos(2)), ...%最大迭代次数
%         'ExecutionEnvironment' ,'cpu',...
%         'GradientThreshold',1, ...
%         'InitialLearnRate',Best_pos(3), ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
%         'LearnRateDropFactor',0.1, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
%         'L2Regularization',0.0001,...
%         'Verbose',0);
%     % 优化算法：使用Adam优化算法（'adam'）。
%     % 最大迭代次数：MaxEpochs 设置为 Best_pos(2) 的四舍五入值，即训练过程中的最大迭代次数由 Best_pos(2) 决定。
%     % 执行环境：ExecutionEnvironment 设置为 'cpu'，表示训练过程将在CPU上执行，不使用GPU加速。
%     % 梯度阈值：GradientThreshold 设置为1，用于裁剪梯度爆炸问题。
%     % 初始学习率：InitialLearnRate 设置为 Best_pos(3)，即初始学习率由 Best_pos(3) 决定。
%     % 学习率调度策略：LearnRateSchedule 设置为 'piecewise'，表示学习率将按照分段函数的方式调整。
%     % 学习率下降周期：LearnRateDropPeriod 设置为 round(0.8*Best_pos(2))，即学习率下降的周期是 Best_pos(2) 的80%的四舍五入值。
%     % 学习率下降因子：LearnRateDropFactor 设置为 0.2，表示在每个下降周期后，学习率将乘以0.2来降低。
%     % L2正则化：L2Regularization 设置为 0.0001，用于防止过拟合。
%     % 详细输出：Verbose 设置为 0，表示在训练过程中不输出详细的训练信息
% 
%     %% start training
%     %  训练
%     net = trainNetwork(Lp_train,targetD',lgraph,options0);
%     % net = trainNetwork(trainD,targetD',lgraph0,options0);
%     %analyzeNetwork(net);% 查看网络结构
%     %  预测
%     t_sim1 = predict(net, Lp_train);
%     t_sim2 = predict(net, Lp_test);
% 
%     %  数据反归一化
%     T_sim1_4 = mapminmax('reverse', t_sim1, ps_output);
%     T_sim2_4 = mapminmax('reverse', t_sim2, ps_output);
% 
%     %  数据格式转换
%     imf_T_sim1(:,uu) = double(T_sim1_4);% cell2mat将cell元胞数组转换为普通数组
%     imf_T_sim2(:,uu) = double(T_sim2_4);
% end
% 
% %% 重构出到真实的测试集与训练集
% [res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);
% % 训练集和测试集划分%% 来自：公众号《淘个代码》
% %% 以下几行代码是为了方便归一化，一般不需要更改！
% T_train_6 = res(1: num_train_s,2);
% T_train_6 = cell2mat(T_train_6)';
% 
% T_test_6 = res(num_train_s+1: end,2);
% T_test_6 = cell2mat(T_test_6)';
% 
% %% 各分量预测的结果相加
% T_sim_a_6 = sum(imf_T_sim1,2);
% T_sim_b_6 = sum(imf_T_sim2,2);
% 
% % % 指标计算
% % disp('…………训练集误差指标…………')
% % [mae11, rmse11, mape11, error11] = calc_error(T_train_6, T_sim_a_6');
% % fprintf('\n')
% % 
% % % -------------------训练集预测效果对比图-------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 调整图形尺寸，设置底色为白色
% % plot(T_train_6, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% % hold on;
% % plot(T_sim_a_6', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% % legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% % title('EMD-SSA-CNN-biLSTM训练集预测效果对比', 'FontSize', 14);
% % xlabel('样本点', 'FontSize', 12);
% % ylabel('发电功率', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% % 
% % disp('…………测试集误差指标…………')
% % [mae12, rmse12, mape12, error12] = calc_error(T_test_6, T_sim_b_6');
% % fprintf('\n')
% % 
% % % -------------------测试集预测效果对比图-------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% % plot(T_test_6, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% % hold on;
% % plot(T_sim_b_6', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% % legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% % title('EM-SSA-CNN-biLSTM测试集预测效果对比', 'FontSize', 14);
% % xlabel('样本点', 'FontSize', 12);
% % ylabel('发电功率', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% % 
% % % -------------------误差曲线图-------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% % plot(T_sim_b_6' - T_test_6, 'LineWidth', 1.5); % 绘制预测误差
% % title('EMD-SSA-CNN-biLSTM误差曲线图', 'FontSize', 14);
% % xlabel('样本点', 'FontSize', 12);
% % ylabel('发电功率误差', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% % % ------------------- Error Metrics Calculation -------------------
% % disp('………… Training Set Error Metrics …………')
% % [mae11, rmse11, mape11, error11] = calc_error(T_train_6, T_sim_a_6');
% % fprintf('\n')
% %%% 
% %%
% % % ------------------- Training Set Prediction Performance Comparison -------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Adjust figure size, set background color to white
% % plot(T_train_6, 'LineWidth', 1.5); % Plot real values, set line width
% % hold on;
% % plot(T_sim_a_6', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
% % legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
% % title('EMD-SSA-CNN-biLSTM Training Set Prediction Performance Comparison', 'FontSize', 14);
% % xlabel('Sample Points', 'FontSize', 12);
% % ylabel('Power Output', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % Set axis font size
% % 
% % disp('………… Test Set Error Metrics …………')
% % [mae12, rmse12, mape12, error12] = calc_error(T_test_6, T_sim_b_6');
% % fprintf('\n')
% % 
% % % ------------------- Test Set Prediction Performance Comparison -------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% % plot(T_test_6, 'LineWidth', 1.5); % Plot real values, set line width
% % hold on;
% % plot(T_sim_b_6', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
% % legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
% % title('EMD-SSA-CNN-biLSTM Test Set Prediction Performance Comparison', 'FontSize', 14);
% % xlabel('Sample Points', 'FontSize', 12);
% % ylabel('Power Output', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % Set axis font size
% % 
% % % ------------------- Error Curve -------------------
% % figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
% % plot(T_sim_b_6' - T_test_6, 'LineWidth', 1.5); % Plot prediction errors
% % title('EMD-SSA-CNN-biLSTM Error Curve', 'FontSize', 14);
% % xlabel('Sample Points', 'FontSize', 12);
% % ylabel('Power Output Error', 'FontSize', 12);
% % grid on;
% % set(gca, 'FontSize', 12); % Set axis font size
% 


%% 无分解
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

% % Error Metrics for Training Set
% disp('………… Training Set Error Metrics …………')
% [mae17, rmse17, mape17, error17] = calc_error(T_train_9, T_sim_a_9');
% fprintf('\n')

% ------------------- Training Set Prediction Performance Comparison -------------------
figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
plot(T_train_9, 'LineWidth', 1.5); % Plot real values, set line width
hold on;
plot(T_sim_a_9', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
title('SSA-CNN-biLSTM Training Set Prediction Performance Comparison', 'FontSize', 14);
xlabel('Sample Points', 'FontSize', 12);
ylabel('Power Output', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12); % Set axis font size

% % Error Metrics for Test Set
disp('………… Test Set Error Metrics …………')
[mae18, rmse18, mape18, error18] = calc_error(T_test_9, T_sim_b_9');
fprintf('\n')

% ------------------- Test Set Prediction Performance Comparison -------------------
figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
plot(T_test_9, 'LineWidth', 1.5); % Plot real values, set line width
hold on;
plot(T_sim_b_9', '--', 'LineWidth', 1.5); % Plot predicted values, set dashed line and line width
legend('True Values', 'Predicted Values', 'FontSize', 12, 'Location', 'best');
title('SSA-CNN-biLSTM Test Set Prediction Performance Comparison', 'FontSize', 14);
xlabel('Sample Points', 'FontSize', 12);
ylabel('Power Output', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12); % Set axis font size

% ------------------- Error Curve -------------------
figure('Position', [200, 300, 900, 300], 'Color', 'w'); % Set background color to white, adjust figure size
plot(T_sim_b_9' - T_test_9, 'LineWidth', 1.5); % Plot prediction errors
title('SSA-CNN-biLSTM Error Curve', 'FontSize', 14);
xlabel('Sample Points', 'FontSize', 12);
ylabel('Power Output Error', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12); % Set axis font size

% 指标计算 
% disp('…………训练集误差指标…………')
% [mae17, rmse17, mape17, error17] = calc_error(T_train_9, T_sim_a_9');
% fprintf('\n')
% 
% % -------------------训练集预测效果对比图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_train_9, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% hold on;
% plot(T_sim_a_9', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% title('SSA-CNN-biLSTM训练集预测效果对比', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% disp('…………测试集误差指标…………')
% [mae18, rmse18, mape18, error18] = calc_error(T_test_9, T_sim_b_9');
% fprintf('\n')
% 
% % -------------------测试集预测效果对比图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_test_9, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% hold on;
% plot(T_sim_b_9', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% title('SSA-CNN-biLSTM预测集预测效果对比', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% % -------------------误差曲线图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_sim_b_9' - T_test_9, 'LineWidth', 1.5); % 绘制预测误差
% title('SSA-CNN-biLSTM-误差曲线图', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率误差', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% 
% 
% % 指标计算
% disp('…………训练集误差指标…………')
% [mae17,rmse17,mape17,error17]=calc_error(T_train_9,T_sim_a_9');
% fprintf('\n')
% 
% figure('Position',[200,300,600,200])
% plot(T_train_9);
% hold on
% plot(T_sim_a_9')
% legend('真实值','预测值')
% title('SSA-CNN-biLSTM训练集预测效果对比')
% xlabel('样本点')
% ylabel('发电功率')
% 
% disp('…………测试集误差指标…………')
% [mae18,rmse18,mape18,error18]=calc_error(T_test_9,T_sim_b_9');
% fprintf('\n')
% 
% 
% figure('Position',[200,300,600,200])
% plot(T_test_9);
% hold on
% plot(T_sim_b_9')
% legend('真实值','预测值')
% title('CEEMDAN-VMD-SSA-CNN-biLSTM预测集预测效果对比')
% xlabel('样本点')
% ylabel('发电功率')
% 
% figure('Position',[200,300,600,200])
% plot(T_sim_b_9'-T_test_9)
% title('CEEMDAN-VMD-SSA-CNN-biLSTM-误差曲线图')
% xlabel('样本点')
% ylabel('发电功率')

%% CEE-VMD(78)
% clc;
% clear
% close all
% X = xlsread('风电场预测.xlsx');
% X = X(5665:8640,:);  %选取3月份数据，最后一列要是预测值哦
% load Co_data.mat
% IMF = Co_data';
% disp('…………………………………………………………………………………………………………………………')
% disp('CEEMDAN-VMD-SSA-CNNbiLSTM预测')
% disp('…………………………………………………………………………………………………………………………')
% 
% %% 对每个分量建模
% for uu=1:size(IMF,2)
%     X_imf=[X(:,1:end-1),IMF(:,uu)];
% 
% 
%     n_in = 5;  % 输入前5个时刻的数据
%     n_out = 1 ; % 此程序为单步预测，因此请将n_out设置为1
%     or_dim = size(X,2) ;       % 记录特征数据维度
%     num_samples = length(X_imf)- n_in; % 样本个数
%     scroll_window = 1;  %如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
%     [res] = data_collation(X_imf, n_in, n_out, or_dim, scroll_window, num_samples);
% 
% 
%     %% 训练集和测试集划分%% 来自：公众号《淘个代码》
% 
%     num_size = 0.8;                              % 训练集占数据集比例  %% 来自：公众号《淘个代码》
%     num_train_s = round(num_size * num_samples); % 训练集样本个数  %% 来自：公众号《淘个代码》
% 
%     %% 以下几行代码是为了方便归一化，一般不需要更改！
%     P_train = res(1: num_train_s,1)
%     P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
%     T_train = res(1: num_train_s,2);
%     T_train = cell2mat(T_train)';
% 
%     P_test = res(num_train_s+1: end,1);
%     P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
%     T_test = res(num_train_s+1: end,2);
%     T_test = cell2mat(T_test)';
% 
% 
%     %  数据归一化
%     [p_train, ps_input] = mapminmax(P_train, 0, 1);
%     p_test = mapminmax('apply', P_test, ps_input);
% 
%     [t_train, ps_output] = mapminmax(T_train, 0, 1);
%     t_test = mapminmax('apply', T_test, ps_output);
% 
%     vp_train = reshape(p_train,n_in,or_dim,num_train_s);
%     vp_test = reshape(p_test,n_in,or_dim,num_samples-num_train_s);
% 
%     vt_train = t_train;
%     vt_test = t_test;
% 
%     f_ = [size(vp_train,1) size(vp_train,2)];
%     outdim = n_out;
% 
%     %%  数据平铺 %% 来自：公众号《淘个代码》
% 
%     for i = 1:size(P_train,2)
%         Lp_train{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
%     end
% 
%     for i = 1:size(p_test,2)
%         Lp_test{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
%     end
% 
% 
% %% 参数设置
%     targetD =  t_train;
%     targetD_test  =  t_test;
% 
%     numFeatures = size(p_train,1);
% %% 小麻雀SSA
% pop=3; % 麻雀数量
% Max_iteration=6; % 最大迭代次数
% dim=3; % 优化lstm的3个参数
% lb = [40,40,0.001];%下边界
% ub = [200,200,0.03];%上边界
% 
% % %归一化后的训练和测试集划分
% xTrain=P_train;
% xTest=P_test;
% yTrain=T_train;
% yTest=T_test;
% numFeatures = size(xTrain,1);
% numResponses = 1;
% fobj = @(x) fun(x,numFeatures,numResponses,xTrain,yTrain,xTest,yTest);%xTrain:归一化后的训练和测试集划分
%  % [Best_score,Best_pos,curve]=WOA(SearchAgents_no,Max_iteration,lb ,ub,dim,finess);
% [Best_pos,Best_score,curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化SSA
% % 
% best_hd  = round(Best_pos(2)); % 最佳隐藏层节点数
% best_lr= round(Best_pos(1));% 最佳初始学习率
% best_l2 = round(Best_pos(3));% 最佳L2正则化系数
% 
% 
% 
%     %% ATTENTION
%     lgraph = layerGraph();                                                 % 建立空白网络结构
%  % 输入特征
% tempLayers = [
%     sequenceInputLayer([numFeatures,1,1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
%     sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
%   % CNN特征提取
% tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
% lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     reluLayer("Name", "relu_1")                                        % 激活层
%     convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
%     reluLayer("Name", "relu_2")];                                      % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% % 池化层
% tempLayers = [
%     globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
%     fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
%     reluLayer("Name", "relu_3")                                        % 激活层
%     fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
%     sigmoidLayer("Name", "sigmoid")];                                  % 激活层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% tempLayers = [
%     sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
%     flattenLayer("Name", "flatten")                                    % 网络铺平层
%      bilstmLayer(6, "Name", "lstm", "OutputMode", "last")                 % bilstm层//（比LSTM多）不同点
% 
%     fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
%     regressionLayer("Name", "regressionoutput")];                      % 回归层
% lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 
% lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
% lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
%                                                                        % 折叠层输出 连接 反折叠层输入  
% lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
% lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
% lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
% lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
% lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出
% 
% 
%     %% 参数设置SSA
%     % 指定训练选项SSA
%     options0 = trainingOptions('adam', ...
%         'MaxEpochs',round(Best_pos(2)), ...%最大迭代次数
%         'ExecutionEnvironment' ,'cpu',...
%         'GradientThreshold',1, ...
%         'InitialLearnRate',Best_pos(3), ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropPeriod',round(0.8*Best_pos(2)), ...
%         'LearnRateDropFactor',0.1, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
%         'L2Regularization',0.0001,...
%         'Verbose',0);
%     % 优化算法：使用Adam优化算法（'adam'）。
%     % 最大迭代次数：MaxEpochs 设置为 Best_pos(2) 的四舍五入值，即训练过程中的最大迭代次数由 Best_pos(2) 决定。
%     % 执行环境：ExecutionEnvironment 设置为 'cpu'，表示训练过程将在CPU上执行，不使用GPU加速。
%     % 梯度阈值：GradientThreshold 设置为1，用于裁剪梯度爆炸问题。
%     % 初始学习率：InitialLearnRate 设置为 Best_pos(3)，即初始学习率由 Best_pos(3) 决定。
%     % 学习率调度策略：LearnRateSchedule 设置为 'piecewise'，表示学习率将按照分段函数的方式调整。
%     % 学习率下降周期：LearnRateDropPeriod 设置为 round(0.8*Best_pos(2))，即学习率下降的周期是 Best_pos(2) 的80%的四舍五入值。
%     % 学习率下降因子：LearnRateDropFactor 设置为 0.2，表示在每个下降周期后，学习率将乘以0.2来降低。
%     % L2正则化：L2Regularization 设置为 0.0001，用于防止过拟合。
%     % 详细输出：Verbose 设置为 0，表示在训练过程中不输出详细的训练信息
% 
%     %% start training
%     %  训练
%     net = trainNetwork(Lp_train,targetD',lgraph,options0);
%     % net = trainNetwork(trainD,targetD',lgraph0,options0);
%     %analyzeNetwork(net);% 查看网络结构
%     %  预测
%     t_sim1 = predict(net, Lp_train);
%     t_sim2 = predict(net, Lp_test);
% 
%     %  数据反归一化
%     T_sim1_4 = mapminmax('reverse', t_sim1, ps_output);
%     T_sim2_4 = mapminmax('reverse', t_sim2, ps_output);
% 
%     %  数据格式转换
%     imf_T_sim1(:,uu) = double(T_sim1_4);% cell2mat将cell元胞数组转换为普通数组
%     imf_T_sim2(:,uu) = double(T_sim2_4);
% end
% 
% %% 重构出到真实的测试集与训练集
% [res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);
% % 训练集和测试集划分%% 来自：公众号《淘个代码》
% %% 以下几行代码是为了方便归一化，一般不需要更改！
% T_train_4 = res(1: num_train_s,2);
% T_train_4 = cell2mat(T_train_4)';
% 
% T_test_4 = res(num_train_s+1: end,2);
% T_test_4 = cell2mat(T_test_4)';
% 
% %% 各分量预测的结果相加
% T_sim_a_4 = sum(imf_T_sim1,2);
% T_sim_b_4 = sum(imf_T_sim2,2);
% 
% % 指标计算
% disp('…………训练集误差指标…………')
% [mae7, rmse7, mape7, error7] = calc_error(T_train_4, T_sim_a_4');
% fprintf('\n')
% 
% % -------------------训练集预测效果对比图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_train_4, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% hold on;
% plot(T_sim_a_4', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% title('CEEMDAN-VMD-SSA-CNN-biLSTM-attention训练集预测效果对比', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% disp('…………测试集误差指标…………')
% [mae8, rmse8, mape8, error8] = calc_error(T_test_4, T_sim_b_4');
% fprintf('\n')
% 
% % -------------------测试集预测效果对比图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_test_4, 'LineWidth', 1.5); % 绘制真实值，设置线宽
% hold on;
% plot(T_sim_b_4', '--', 'LineWidth', 1.5); % 绘制预测值，设置虚线和线宽
% legend('真实值', '预测值', 'FontSize', 12, 'Location', 'best');
% title('CEEMDAN-VMD-SSA-CNN-biLSTM-attention预测集预测效果对比', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小
% 
% % -------------------误差曲线图-------------------
% figure('Position', [200, 300, 900, 300], 'Color', 'w'); % 设置底色为白色，调整图形尺寸
% plot(T_sim_b_4' - T_test_4, 'LineWidth', 1.5); % 绘制预测误差
% title('CEEMDAN-VMD-SSA-CNN-biLSTM-attention误差曲线图', 'FontSize', 14);
% xlabel('样本点', 'FontSize', 12);
% ylabel('发电功率误差', 'FontSize', 12);
% grid on;
% set(gca, 'FontSize', 12); % 设置坐标轴字体大小

% % 指标计算
% disp('…………训练集误差指标…………')
% [mae7,rmse7,mape7,error7]=calc_error(T_train_4,T_sim_a_4');
% fprintf('\n')
% 
% figure('Position',[200,300,600,200])
% plot(T_train_4);
% hold on
% plot(T_sim_a_4')
% legend('真实值','预测值')
% title('CEEMDAN-VMD-SSA-CNN-biLSTM训练集预测效果对比')
% xlabel('样本点')
% ylabel('发电功率')
% 
% disp('…………测试集误差指标…………')
% [mae8,rmse8,mape8,error8]=calc_error(T_test_4,T_sim_b_4');
% fprintf('\n')
% 
% 
% figure('Position',[200,300,600,200])
% plot(T_test_4);
% hold on
% plot(T_sim_b_4')
% legend('真实值','预测值')
% title('CEEMDAN-VMD-SSA-CNN-biLSTM预测集预测效果对比')
% xlabel('样本点')
% ylabel('发电功率')
% 
% figure('Position',[200,300,600,200])
% plot(T_sim_b_4'-T_test_4)
% title('CEEMDAN-VMD-SSA-CNN-biLSTM-误差曲线图')
% xlabel('样本点')
% ylabel('发电功率')
% 
% %%
% %%
% %%
% %% 三个对比
% 
% figure
% set(0,'defaultfigurecolor','w') ;
% plot(T_test_4,'k','linewidth',1);
% hold on;
% plot(T_sim_b_4','b','linewidth',1);
% hold on;
% plot(T_sim_b_7','g','linewidth',1);
% hold on;
% plot(T_sim_b_8','r','linewidth',1);
% hold on;
% plot(T_sim_b_9','Color', [0.5, 0.2, 0.8],'linewidth',1);
% legend('Target','CEEMDAN-VMD-CNN-SSA-BILSTM-attention','CEEMDAN-CNN-SSA-BILSTM-attention','VMD-CNN-SSA-BILSTM-attention','CNN-SSA-BILSTM-attention');
% title('三种模型测试集结果对比图');
% xlabel('Sample Index');
% xlabel('Wind Speed');
% grid on;
% 
% %% 线形对比更加明显
% figure
% set(0,'defaultfigurecolor','w');
% plot(T_test_4, 'k', 'linewidth', 1);  % 黑色实线
% hold on;
% plot(T_sim_b_4, 'b--', 'linewidth', 1);  % 蓝色虚线
% hold on;
% plot(T_sim_b_7, 'g--', 'linewidth', 1);  % 绿色虚线
% hold on;
% plot(T_sim_b_8, 'r--', 'linewidth', 1);  % 红色虚线
% hold on;
% plot(T_sim_b_9, 'Color', [0.7, 0.1, 0.9], 'linewidth', 2);  % 更加突出的紫色实线
% legend('Target','CEEMDAN-VMD-CNN','CEEMDAN-VMD-BILSTM','CEEMDAN-VMD-CNN-BILSTM','CEEMDAN-VMD-SSA-CNN-BILSTM-attention');
% title('三种模型测试集结果对比图');
% xlabel('Sample Index');
% ylabel('Wind Speed');  % 将第二个 xlabel 改为 ylabel
% grid on;
% %% 
% figure
% set(0,'defaultfigurecolor','w');
% plot(error8, 'k-', 'linewidth', 1.5, 'Marker', 'o', 'DisplayName', 'CEEMDAN-VMD-SSA-CNN-BILSTM-attention-Error');
% hold on;
% plot(error14, 'g--', 'linewidth', 1.5, 'Marker', 'S', 'DisplayName', 'CEEMDAN-SSA-CNN-BILSTM-attention-Error');
% plot(error16, 'r-.', 'linewidth', 1.5, 'Marker', '*', 'DisplayName', 'VMD-SSA-CNN-BILSTM-attention-Error');
% plot(error18, 'b:', 'linewidth', 1.5, 'Marker', '+', 'DisplayName', 'SSA-CNN-BILSTM-ATTENTION-Error');
% legend('Location', 'bestoutside');
% title('模型误差对比图', 'FontSize', 14);
% xlabel('Sample Index', 'FontSize', 12);
% ylabel('Error Value', 'FontSize', 12);
% grid on;
% 
