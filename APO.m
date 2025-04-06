
function [BestX,BestF,curve]=APO(N,T,lb,ub,dim,fobj)

PopPos=zeros(N,dim);
PopFit=zeros(N,1);

for i=1:N
    PopPos(i,:)=rand(1,dim).*(ub-lb)+lb;
    PopFit(i)=fobj(PopPos(i,:));
end

BestF=inf;
BestX=[];

for i=1:N
    if PopFit(i)<=BestF
        BestF=PopFit(i);
        BestX=PopPos(i,:);
    end
end

curve=zeros(T,1);
%% -------------------Start iteration------------------------------------%
for It=1:T
    for i=1:N

        theta1=(1-It/T);
        B=2*log(1/rand)*theta1;
%% -------------------1.Aerial Flight Stage-------------------%
        if B > 0.5

            while true
                K = [1:i-1, i+1:N];
                RandInd = K(randi([1,N-1]));
                step1 = PopPos(i, :) - PopPos(RandInd, :);
                if norm(step1) ~= 0 && ~isequal(PopPos(i, :), PopPos(RandInd, :))
                    break;
                end
            end
             %% -------------------1.1 Aerial search.-------------------%
            Y=PopPos(i,:) +  Levy(dim) .* step1 +round(0.5*(0.05+rand))*randn;

             %% -------------------1.2 Swooping predation.-------------------%
            R=rand(1,dim);
            step2=(R-0.5)*pi;
            S=tan(step2);
            Z=Y.*S;

            Y = SpaceBound(Y, ub, lb);
            Z = SpaceBound(Z, ub, lb);
            NewPop=[Y;Z];
            NewPopfit=[fobj(Y);fobj(Z)];
            [~,sorted_indexes]=sort(NewPopfit);
            newPopPos=NewPop(sorted_indexes(1),:);

        else
 %% -------------------2.Underwater Foraging Stage-------------------%
            F=0.5;
            K = [1:i - 1, i + 1:N];
            RandInd = K(randi([1, N - 1], 1, 3));
            f=(0.1*(rand-1)*(T-It))/T;
            while true
                RandInd = K(randi([1 N-1], 1, 3));
                step1 = PopPos(RandInd(2), :) - PopPos(RandInd(3), :);
                               
                if norm(step1) ~= 0 && RandInd(2) ~= RandInd(3)
                    break;
                end
            end
     
            %% -------------------2.1	Gathering foraging-------------------%
            if rand<0.5
                W = PopPos(RandInd(1), :) + F .* step1;
            else
                W = PopPos(RandInd(1), :) + F .*Levy(dim).* step1;
            end

            %% -------------------2.2Intensifying search-------------------%
            Y=(1+f)*  W ;

            %% -------------------2.3Underwater Foraging Stage-------------------%
            while true
                rand_leader_index1 = floor(N * rand() + 1);
                rand_leader_index2 = floor(N * rand() + 1);
                X_rand1 = PopPos(rand_leader_index1, :);
                X_rand2 = PopPos(rand_leader_index2, :);
                step2 = X_rand1 - X_rand2;

                if norm(step2) ~= 0 && ~isequal(X_rand1, X_rand2)
                    break;
                end
            end

            Epsilon = unifrnd(0, 1);
            if rand<0.5
                Z = PopPos(i, :) + Epsilon .* step2; % Eq.(11)4.3
            else
                Z = PopPos(i, :) + F .* Levy(dim) .* step2;
            end


            NewPop=[W;Y;Z];
            NewPopfit=[fobj(W);fobj(Y);fobj(Z)];
            [~,sorted_indexes]=sort(NewPopfit);
            newPopPos=NewPop(sorted_indexes(1),:);

        end

        newPopPos = SpaceBound(newPopPos, ub, lb);
        newPopFit = fobj(newPopPos);

        if newPopFit < PopFit(i)
            PopFit(i) = newPopFit;
            PopPos(i, :) = newPopPos;
        end
    end

    for i=1:N
        if PopFit(i)<BestF
            BestF=PopFit(i);
            BestX=PopPos(i,:);
        end
    end

    curve(It)=BestF;

end
end

function o=Levy(Dim)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,Dim)*sigma;
v=randn(1,Dim);
step=u./abs(v).^(1/beta);
o=step;
end

function  X=SpaceBound(X,Up,Low)


    Dim=length(X);
    S=(X>Up)+(X<Low);    
    X=(rand(1,Dim).*(Up-Low)+Low).*S+X.*(~S);
end

%%  闲鱼：深度学习与智能算法
%%  唯一官方店铺：https://mbd.pub/o/autAPOr-aWWbm3BtZw==
%%  微信公众号：强盛机器学习，关注公众号获得更多免费代码！