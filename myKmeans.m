function solution = myKmeans(X, K, nbr_runs, nbr_iter_max, verbose)
%   function res = myKmeans(X, K, nbr_runs, nbr_iter_max, verbose)
%
%   Algorithme des K-means
%
%
%
% Faicel CHAMROUKHI Septembre 2008 (mise a jour)
%
%
%
% distance euclidienne
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<5; verbose=0;end
if nargin<4; nbr_iter_max = 300;end
if nargin<3; nbr_runs = 20;end


[n, p] = size(X);
% if one class
global_mean = mean(X,1);

if K==1
    dmin = sum((X-ones(n, 1)*global_mean).^2,2);
    solution.muk = global_mean;
    klas = ones(n,1);
    solution.klas = klas;
    solution.err = sum(dmin);
    solution.Zik = ones(n,1);
    return;
end


nbr_run = 0;
best_solution.err = inf;
while (nbr_run<nbr_runs)
    nbr_run=nbr_run + 1;
    if (nbr_runs>1 && verbose); fprintf('Kmeans run n° : %d  \n',nbr_run);end
    
    iter = 0;
    converged = 0;
    previous_err = -inf;
    Zik = zeros(n,K);% partition
    %% 1. Initialization of the centres
    rnd_indx = randperm(n);
    centres = X(rnd_indx(1:K),:);
    while (iter<nbr_iter_max && ~converged)
        iter = iter+1;
        old_centres = centres;
        
        % The Euclidean distances
        eucld_dist = zeros(n, K);
        for k = 1:K
            muk = centres(k,:);
            eucld_dist(:,k) = sum((X-ones(n,1)*muk).^2,2);
        end
        %% classification step
        
        [dmin, klas] = min(eucld_dist,[],2);
        
        Zik = (klas*ones(1,K))==(ones(n,1)*[1:K]);
        %% relocation step
        for k=1:K
            ind_ck = find(klas==k);
            %if empty classes
            if isempty(ind_ck)
                centres(k,:)= old_centres(k,:);
            else
                % update the centres
                centres(k,:) = mean(X(ind_ck,:),1);
            end
        end
        
        % test of convergence
        current_err = sum(sum(Zik.*eucld_dist,2)); % the distorsion measure

        if (abs(current_err-previous_err))/previous_err <1e-6
            converged = 1;
        end
        previous_err = current_err;
        if verbose
            fprintf('Kmeans : Iteration  %d  Objective: %6f  \n', iter, current_err);
        end
        solution.stored_err(iter) = current_err;
    end% one run
    %%
    solution.muk = centres;
    solution.Zik = Zik;
    solution.klas = klas;
    solution.err = current_err;
    %
    if current_err < best_solution.err
        best_solution = solution;
    end
end %en of the Kmeans runs

solution = best_solution;


