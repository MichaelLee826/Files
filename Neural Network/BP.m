clear all;
close all;
SamNum = 4995;                           %训练样本数
TestSamNum = 495;                        %测试样本数
HiddenUnitNum = 20;                      %隐节点数 5-14
InDim = 20;                              %输入维数 
OutDim = 1;                              %输出维数

%样本输入输出
rand('state',sum(100*clock))
N = importdata('C:\Users\Dazzal\Documents\Tencent Files\175760063\FileRecv\机器学习大作业\多变量时间序列预测\train_X.txt');
Y1 = importdata('C:\Users\Dazzal\Documents\Tencent Files\175760063\FileRecv\机器学习大作业\多变量时间序列预测\train_y.txt');
X1 = N(:,1);
X2 = N(:,2);
X3 = N(:,3);


y_output = Y1(6:5000);
y_i1 = Y1(5:4999);
y_i2 = Y1(4:4998);
y_i3 = Y1(3:4997);
y_i4 = Y1(2:4996);
y_i5 = Y1(1:4995);

x1_i1 = X1(5:4999);
x1_i2 = X1(4:4998);
x1_i3 = X1(3:4997);
x1_i4 = X1(2:4996);
x1_i5 = X1(1:4995);

x2_i1 = X2(5:4999);
x2_i2 = X2(4:4998);
x2_i3 = X2(3:4997);
x2_i4 = X2(2:4996);
x2_i5 = X2(1:4995);

x3_i1 = X3(5:4999);
x3_i2 = X3(4:4998);
x3_i3 = X3(3:4997);
x3_i4 = X3(2:4996);
x3_i5 = X3(1:4995);

SamIn = [y_i1,y_i2,y_i3,y_i4,y_i5,x1_i1,x1_i2,x1_i3,x1_i4,x1_i5,x2_i1,x2_i2,x2_i3,x2_i4,x2_i5,x3_i1,x3_i2,x3_i3,x3_i4,x3_i5]';
SamOut = y_output';
%测试输入
x_test = importdata('C:\Users\Dazzal\Desktop\神经网络\机器学习大作业\多变量时间序列预测\test_X.txt');
y_test = importdata('C:\Users\Dazzal\Desktop\神经网络\机器学习大作业\多变量时间序列预测\test_y.txt');
x1_test = x_test(:,1);
x2_test = x_test(:,2);
x3_test = x_test(:,3);
x4_test = x_test(:,4);


y_t1 = Y1(5:499);
y_t2 = Y1(4:498);
y_t3 = Y1(3:497);
y_t4 = Y1(2:496);
y_t5 = Y1(1:495);

x1_t1 = X1(5:499);
x1_t2 = X1(4:498);
x1_t3 = X1(3:497);
x1_t4 = X1(2:496);
x1_t5 = X1(1:495);

x2_t1 = X2(5:499);
x2_t2 = X2(4:498);
x2_t3 = X2(3:497);
x2_t4 = X2(2:496);
x2_t5 = X2(1:495);

x3_t1 = X3(5:499);
x3_t2 = X3(4:498);
x3_t3 = X3(3:497);
x3_t4 = X3(2:496);
x3_t5 = X3(1:495);
TestSamIn = [y_t1,y_t2,y_t3,y_t4,y_t5,x1_t1,x1_t2,x1_t3,x1_t4,x1_t5,x2_t1,x2_t2,x2_t3,x2_t4,x2_t5,x3_t1,x3_t2,x3_t3,x3_t4,x3_t5]';
TestSamOut = y_test';

MaxEpochs = 20000;          %训练次数
lr = 0.00021;                 %学习率
E0 = 0.005;                   %目标误差

W1 = 0.2*rand(20,20)-0.1;
B1 = 0.2*rand(20,1)-0.1;
W2 = 0.2*rand(1,20)-0.1;
B2 = 0.2*rand(1,1)-0.1;

W1Ex = [W1 B1];
W2Ex = [W2 B2];

SamInEx = [SamIn' ones(SamNum,1)]';
ErrHistory = [];
for i=1:MaxEpochs
    %正向传播计算网络输出
    HiddenOut = logsig(W1Ex*SamInEx);
    HiddenOutEx = [HiddenOut' ones(SamNum,1)]';
    NetworkOut = W2Ex * HiddenOutEx;
    
    Error = SamOut-NetworkOut;
    SSE = sumsqr(Error);
    
    ErrHistory = [ErrHistory SSE];
    
    if SSE<E0,break,end
    
    %计算反传误差
    Delta2 = 0.5*Error;
    Delta1 = W2'*Delta2.*HiddenOut.*(1-HiddenOut);
    
    %计算权值调节量
    dW2Ex = Delta2*HiddenOutEx';
    dW1Ex = Delta1*SamInEx';
    
    %权值调节
    W1Ex = W1Ex + lr*dW1Ex;
    W2Ex = W2Ex + lr*dW2Ex;
    
    %分离层到输出层的初始权值
    W2 = W2Ex(:,1:HiddenUnitNum);
end

W1 = W1Ex(:,1:InDim)
B1 = W1Ex(:,InDim +1 )
W2
B2 = W2Ex(:,1+HiddenUnitNum);

%测试
T = 1:500;
TestHiddenOut = logsig(W1*TestSamIn+repmat(B1,1,TestSamNum));
TestNNOut = W2*TestHiddenOut+repmat(B2,1,TestSamNum);
TestNNOut = [-1.8783,-1.6984,-2.554,-2.3422,-2.0212,TestNNOut];
plot (T,TestNNOut);
hold on
plot (T,TestSamOut,'r');
sum1 = 0;
for count = 1:500
    sum1 = sum1 + (TestNNOut(count)-TestSamOut(count))*(TestNNOut(count)-TestSamOut(count));
end

RMSE = sqrt(sum1/500);

sum2 = 0;
for count = 1:500
    sum2 = sum2 + abs(TestNNOut(count)-TestSamOut(count));
end

MAE = sum2/500;

%绘制学习误差曲线
figure
hold on
grid
[xx,Num] = size(ErrHistory);
plot(1:Num,ErrHistory,'k-');
    
    


