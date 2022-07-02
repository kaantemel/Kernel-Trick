clc
clear


filename = 'Social_Network_Ads.csv';

A = readtable(filename);
idx1 = fsrftest(A,A.UserID);
idx2 = fsrftest(A,A.Age);
idx3 = fsrftest(A,A.EstimatedSalary);
idx = [idx1;idx2;idx3];

%ranking the features using fsrftest

S = sum(idx);
M = mean(S);
ST = std(S);
f = find(S>(M+ST));


%using curve to find which features should be extracted

count=0;
for i = f
    A(:,i-count)=[];
    count=count+1;
end

A.UserID = [];

%standardizing our data and applying Naive Bayes Classifier


stand_1 = (A.Age - mean(A.Age))/std(A.Age);
A.Age = stand_1; 
stand_2=(A.EstimatedSalary - mean(A.EstimatedSalary))/std(A.EstimatedSalary);
A.EstimatedSalary=stand_2; 

classified = fitcnb(A,'Purchased');

%Cross validation

CVmodel = crossval(classified);
loss = kfoldLoss(CVmodel)

% Visualization part of the NB classification

a=min(A.Age):0.01:max(A.Age);
b=min(A.EstimatedSalary):0.01:max(A.EstimatedSalary);
[c d]=meshgrid(a,b);
x=[c(:) d(:)];
e=predict(classified,x);
gscatter(c(:),d(:),e,'cym');
hold on;
gscatter(A.Age,A.EstimatedSalary,A.Purchased,'rg','.',30);
title('Naive_Bayesain Classification Visualization');









