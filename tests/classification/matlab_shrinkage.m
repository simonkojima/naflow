clearvars

rng(42);

if ispc
    home_dir = winqueryreg('HKEY_CURRENT_USER',...
        ['Software\Microsoft\Windows\CurrentVersion\' ...
         'Explorer\Shell Folders'],'Personal');
else
    home_dir = char(java.lang.System.getProperty('user.home'));
end

run(fullfile(home_dir, "git", "bbci_public", "startup_bbci_toolbox.m"))

mu{1} = [6, 2];
sigma{1} = [1, 0.8; 0.8, 1];

mu{2} = [2, 8];
sigma{2} = [1, 0.8; 0.8, 1];

N{1} = 1000;
N{2} = 1000;

data{1} = mvnrnd(mu{1}, sigma{1}, N{1});
data{2} = mvnrnd(mu{2}, sigma{2}, N{2});

if 0
    figure()
    scatter(data{1}(:, 1), data{1}(:, 2), 'filled', 'r')
    hold on
    scatter(data{2}(:, 1), data{2}(:, 2), 'filled', 'b')
    axis equal
end

xTr = cat(1, data{1}, data{2});
yTr = cat(1, ones(N{1}, 1), zeros(N{2}, 1));
yTr = cat(2, yTr, ~yTr);

% % Plain LDA
C = train_LDA(xTr', yTr', 'scaling', true, 'StoreCov', true, 'StoreInvcov', true)
gamma = C.gamma;
w = C.w;
b = C.b;
cov = C.cov;
invcov = C.invcov;
save('plain_lda.mat', 'xTr', 'yTr', 'gamma', 'w', 'b', 'cov', 'invcov', '-v6')

% Shrinkage LDA
C = train_RLDAshrink(xTr', yTr', 'scaling', true, 'StoreCov', true, 'StoreInvcov', true)
gamma = C.gamma;
w = C.w;
b = C.b;
cov = C.cov;
invcov = C.invcov;
save('shrinkage_lda.mat', 'xTr', 'yTr', 'gamma', 'w', 'b', 'cov', 'invcov', '-v6')