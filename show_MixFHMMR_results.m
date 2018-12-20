function show_MixFHMMR_results(data, mixFHMMR)
set(0,'defaultaxesfontsize',14);
[n, m] = size(data);

t = 0:m-1;

K = length(mixFHMMR.model.w_k);
colors = {'r','b','g','m','c','k','y'};
% colors_cluster_means = {'m','c','k','b','r','g'};
colors_cluster_means = {[0.8 0 0],[0 0 0.8],[0 0.8 0],'m','c','k','y'};

%% 
scrsz = get(0,'ScreenSize');
figure('Position',[10 scrsz(4)/2 550 scrsz(4)/2.15]);
plot(t,data')
xlabel('t')
ylabel('y(t)')
title('original time series')

%
% clustered data and cluster meands
figure('Position',[scrsz(4) scrsz(4)/2 550 scrsz(4)/2.15]);
for k=1:K
    cluster_k = data(mixFHMMR.stats.klas==k,:);
    plot(t,cluster_k','color',colors{k},'linewidth',0.001);    
    hold on
%     plot(t,solution.smoothed(:,k),'color',colors{k},'linewidth',3)
    plot(t,mixFHMMR.stats.smoothed(:,k),'color',colors_cluster_means{k},'linewidth',3)
end
title('Clustered time series')
ylabel('y(t)')
xlabel('t')

%% clusters and cluster means
for k=1:K
    cluster_k = data(mixFHMMR.stats.klas==k,:); 
    
    figure, plot(t,cluster_k','color',colors{k})
    hold on, plot(t,mixFHMMR.stats.smoothed(:,k),'color',colors_cluster_means{k},'linewidth',3)
    title(['Cluster ',int2str(k)])
    ylabel('y(t)');
    xlabel('t'); 
end 