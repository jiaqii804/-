% 淘个代码 %%
%2023/06/15 %
%微信公众号搜索：淘个代码
%
clc
clear
close
addpath('CEEMDAN(完全自适应噪声集合经验模态分解)\')
fs=4;%采样频率
Ts=1/fs;%采样周期
STA=1; %采样起始位置
%----------------导入风电场的数据-----------------------------------------
X = xlsread('风电场预测.xlsx');
X = X(5665:8640,end);  %选取3月份数据，最后一列要是预测值哦
L=length(X);%采样点数
t=(0:L-1)*Ts;%时间序列
%% CEEMDAN分解
Nstd = 0.2;
NR = 500;
MaxIter = 5000;
[modes,~]=ceemdan(X',0.2,500,5000);


%% 绘图
figure('Position',[100,10,300,700]);
imfn=modes;
n=size(imfn,1);
subplot(n+1,1,1);
plot(t,X); %故障信号
ylabel('原始信号','fontsize',12,'fontname','宋体');

for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,modes(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
    ylabel(['IMF' int2str(n1)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
end
xlabel('时间\itt/h','fontsize',12,'fontname','宋体');

%% 计算样本熵
dim = 2;   %   dim：嵌入维数(一般取1或者2)
tau = 1;   %下采样延迟时间（在默认值为1的情况下，用户可以忽略此项）
for i = 1:n
	x=modes(i,:);%
    r = 0.2*std(x);  %   r：相似容限( 通常取0.1*Std(data)~0.25*Std(data) )
    Sample_Entropy(i,:) = SampleEntropy( dim, r, x, tau );
end

%% 根据样本熵进行kmeans聚类
%使用matlab自带k-means聚类
[idx, c] = kmeans(Sample_Entropy, 3); % 聚成3类,
% 观察idx的前几个数字，该数字所在的行即为高频分量。
Co_IMF1 = sum(modes(find(idx==2),:));  %高频分量
Co_IMF2 = sum(modes(find(idx==3),:));   %中频分量
Co_IMF3 = sum(modes(find(idx==1),:));   %低频分量

figure
subplot(3,1,1);
plot(Co_IMF1);ylabel('Co-IMF1');hold on
title('多个IMF分量的K-means聚类结果')
subplot(3,1,2);
plot(Co_IMF2);ylabel('Co-IMF2');hold on
subplot(3,1,3);
plot(Co_IMF3);ylabel('Co-IMF3');hold on

%% 调用VMD对高频分量Co-IMF1分解

K = 3;
vmddata = vmd(Co_IMF1,'NumIMFs',K,'PenaltyFactor',2500);
Co_data = [vmddata';Co_IMF2;Co_IMF3]; %合并VMD分解高频分量与Co_IMF2;Co_IMF3分量
save Co_data.mat Co_data

%% VMD分解的高频分量与Co_IMF2;Co_IMF3分量绘制到一张图上
figure;
imfn=Co_data;
n=size(imfn,1);
subplot(n+1,1,1);
plot(t,X); %故障信号
ylabel('原始信号','fontsize',12,'fontname','宋体');

for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,imfn(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
    ylabel(['IMF' int2str(n1)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
end
xlabel('时间\itt/h','fontsize',12,'fontname','宋体');


% 频谱图
figure('Name','频谱图','Color','white');
for i = 1:n
    p=abs(fft(Co_data(i,:)));
    subplot(n,1,i);
    plot((0:L-1)*fs/L,p)
    xlim([0 fs/2])
    if i ==1
        title('频谱图'); xlabel('频率'); ylabel(['IMF' int2str(i)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
    else
        xlabel('频率');  ylabel(['IMF' int2str(i)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
    end
end
set(gcf,'color','w');

