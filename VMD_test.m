% test-script for VMD
% authors: Dominique Zosso and Konstantin Dragomiretskiy
% zosso@math.ucla.edu --- http://www.math.ucla.edu/~zosso
% Initial release 2013-12-12 (c) 2013
%
% When using this code, please do cite our paper:
% -----------------------------------------------
% K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
% on Signal Processing (in press)
% please check here for update reference: 
%          http://dx.doi.org/10.1109/TSP.2013.2288675

%% 
% tic
% clc
% clear all
% fs=1;%采样频率，即时间序列两个数据之间的时间间隔，这里间隔1h采样
% Ts=1/fs;%采样周期
% L=701;%采样点数,即有多少个数据
% t=(0:L-1)*Ts;%时间序列
% STA=0; %采样起始位置，这里第0h开始采样
% 
% X = xlsread('风电场预测.xlsx');
% X = X(5665:8640,end);
% 
% %--------- some sample parameters forVMD：对于VMD样品参数进行设置---------------
% alpha = 2500;       % moderate bandwidth constraint：适度的带宽约束/惩罚因子
% tau = 0;          % noise-tolerance (no strict fidelity enforcement)：噪声容限（没有严格的保真度执行）
% K = 4;              % modes：分解的模态数
% DC = 0;             % no DC part imposed：无直流部分
% init = 1;           % initialize omegas uniformly  ：omegas的均匀初始化
% tol = 1e-7         
% %--------------- Run actual VMD code:数据进行vmd分解---------------------------
% [u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);
% 
% save vmd_data u
% 
% figure(1);
% imfn=u;
% n=size(imfn,1); %size(X,1),返回矩阵X的行数；size(X,2),返回矩阵X的列数；N=size(X,2)，就是把矩阵X的列数赋值给N
% subplot(n+1,1,1);  % m代表行，n代表列，p代表的这个图形画在第几行、第几列。例如subplot(2,2,[1,2])
% plot(t,X); %故障信号
% ylabel('原始信号','fontsize',12,'fontname','宋体');
% 
% for n1=1:n
%     subplot(n+1,1,n1+1);
%     plot(t,u(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
%     ylabel(['IMF' int2str(n1)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
% end
%  xlabel('时间\itt/hour','fontsize',12,'fontname','宋体');
%  toc;
%  %----------------------计算中心频率确定分解个数K-----------------------------
% average=mean(omega);%求矩阵列的平均值
tic
clc
clear all

% 采样参数
fs = 1; % 采样频率，即时间序列两个数据之间的时间间隔，这里间隔1h采样
Ts = 1 / fs; % 采样周期
L = 701; % 采样点数,即有多少个数据
STA = 0; % 采样起始位置，这里第0h开始采样

% 读取数据
X = xlsread('风电场预测.xlsx');
X = X(5665:8640, end); % 选择3月份数据

% 确保时间序列长度与数据长度一致
numDataPoints = length(X);
t = (0:numDataPoints-1) * Ts; % 更新时间序列

% VMD参数设置
alpha = 2500;       % moderate bandwidth constraint：适度的带宽约束/惩罚因子
tau = 0;            % noise-tolerance (no strict fidelity enforcement)：噪声容限（没有严格的保真度执行）
K = 4;              % modes：分解的模态数
DC = 0;             % no DC part imposed：无直流部分
init = 1;           % initialize omegas uniformly  ：omegas的均匀初始化
tol = 1e-7;         % 收敛容限

% 数据进行VMD分解
[u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);

save vmd1_data u

% 绘制图形
figure(1);
imfn = u;
n = size(imfn, 1); % IMF分量的数量
subplot(n + 1, 1, 1);  % 设置图形为 n + 1 行，1 列，第一位置绘制原始信号
plot(t, X); % 原始信号
ylabel('原始信号', 'fontsize', 12, 'fontname', '宋体');

for n1 = 1:n
    subplot(n + 1, 1, n1 + 1);
    plot(t, u(n1, :)); % 输出IMF分量
    ylabel(['IMF' int2str(n1)]); % y轴命名
end
xlabel('时间\itt/hour', 'fontsize', 12, 'fontname', '宋体');

toc;

% 计算中心频率
average = mean(omega); % 求矩阵列的平均值


%%
%--------------- Preparation
% clear all;
% close all;
% clc;
% 
% % Time Domain 0 to T
% T = 1000;
% fs = 1/T;
% t = (1:T)/T;
% freqs = 2*pi*(t-0.5-1/T)/(fs);
% 
% % center frequencies of components
% f_1 = 2;
% f_2 = 24;
% f_3 = 288;
% 
% % modes
% v_1 = (cos(2*pi*f_1*t));
% v_2 = 1/4*(cos(2*pi*f_2*t));
% v_3 = 1/16*(cos(2*pi*f_3*t));
% 
% % for visualization purposes
% fsub = {};
% wsub = {};
% fsub{1} = v_1;
% fsub{2} = v_2;
% fsub{3} = v_3;
% wsub{1} = 2*pi*f_1;
% wsub{2} = 2*pi*f_2;
% wsub{3} = 2*pi*f_3;
% 
% % composite signal, including noise
% f = v_1 + v_2 + v_3 + 0.1*randn(size(v_1));
% f_hat = fftshift((fft(f)));
% 
% % some sample parameters for VMD
% alpha = 2000;        % moderate bandwidth constraint
% tau = 0;            % noise-tolerance (no strict fidelity enforcement)
% K = 3;              % 3 modes
% DC = 0;             % no DC part imposed
% init = 1;           % initialize omegas uniformly
% tol = 1e-7;
% 
% 
% 
% 
% 
% 
% %--------------- Run actual VMD code
% 
% [u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %--------------- Visualization
% 
% % For convenience here: Order omegas increasingly and reindex u/u_hat
% [~, sortIndex] = sort(omega(end,:));
% omega = omega(:,sortIndex);
% u_hat = u_hat(:,sortIndex);
% u = u(sortIndex,:);
% linestyles = {'b', 'g', 'm', 'c', 'c', 'r', 'k'};
% 
% figure('Name', 'Composite input signal' );
% plot(t,f, 'k');
% set(gca, 'XLim', [0 1]);
% 
% for sub = 1:length(fsub)
%     figure('Name', ['Input signal component ' num2str(sub)] );
%     plot(t,fsub{sub}, 'k');
%     set(gca, 'XLim', [0 1]);
% end
% 
% figure('Name', 'Input signal spectrum' );
% loglog(freqs(T/2+1:end), abs(f_hat(T/2+1:end)), 'k');
% set(gca, 'XLim', [1 T/2]*pi*2, 'XGrid', 'on', 'YGrid', 'on', 'XMinorGrid', 'off', 'YMinorGrid', 'off');
% ylims = get(gca, 'YLim');
% hold on;
% for sub = 1:length(wsub)
%     loglog([wsub{sub} wsub{sub}], ylims, 'k--');
% end
% set(gca, 'YLim', ylims);
% 
% figure('Name', 'Evolution of center frequencies omega');
% for k=1:K
%     semilogx(2*pi/fs*omega(:,k), 1:size(omega,1), linestyles{k});
%     hold on;
% end
% set(gca, 'YLim', [1,size(omega,1)]);
% set(gca, 'XLim', [2*pi,0.5*2*pi/fs], 'XGrid', 'on', 'XMinorGrid', 'on');
% 
% figure('Name', 'Spectral decomposition');
% loglog(freqs(T/2+1:end), abs(f_hat(T/2+1:end)), 'k:');
% set(gca, 'XLim', [1 T/2]*pi*2, 'XGrid', 'on', 'YGrid', 'on', 'XMinorGrid', 'off', 'YMinorGrid', 'off');
% hold on;
% for k = 1:K
%     loglog(freqs(T/2+1:end), abs(u_hat(T/2+1:end,k)), linestyles{k});
% end
% set(gca, 'YLim', ylims);
% 
% 
% for k = 1:K
%     figure('Name', ['Reconstructed mode ' num2str(K)]);
%     plot(t,u(k,:), linestyles{k});   hold on;
%     if ~isempty(fsub)
%         plot(t, fsub{min(k,length(fsub))}, 'k:');
%     end
%     set(gca, 'XLim', [0 1]);
% end
% 
