N = importdata('C:\Users\Dazzal\Documents\Tencent Files\175760063\FileRecv\机器学习大作业\多变量时间序列预测\train_X.txt');
Y1 = importdata('C:\Users\Dazzal\Documents\Tencent Files\175760063\FileRecv\机器学习大作业\多变量时间序列预测\train_y.txt');
X1 = N(:,1);
X2 = N(:,2);
X3 = N(:,3);
X4 = N(:,4);
T = 1:5000;
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

x4_i1 = X4(5:4999);
x4_i2 = X4(4:4998);
x4_i3 = X4(3:4997);
x4_i4 = X4(2:4996);
x4_i5 = X4(1:4995);

X = [ones(length(y_output),1),y_i1,y_i2,y_i3,y_i4,y_i5, x1_i1,x1_i2,x1_i3,x1_i4,x1_i5,  x2_i1,x2_i2,x2_i3,x2_i4,x2_i5, x3_i1,x3_i2,x3_i3,x3_i4,x3_i5, x4_i1,x4_i2,x4_i3,x4_i4,x4_i5];
Y = y_output;
[b,bint,r,rint,stats] = regress(Y,X);

x_test = importdata('C:\Users\Dazzal\Desktop\神经网络\机器学习大作业\多变量时间序列预测\test_X.txt');
y_test = importdata('C:\Users\Dazzal\Desktop\神经网络\机器学习大作业\多变量时间序列预测\test_y.txt');
x1_test = x_test(:,1)';
x2_test = x_test(:,2)';
x3_test = x_test(:,3)';
x4_test = x_test(:,4)';
y_test1 = y_test';


y_tout(1) = -1.8783;
y_tout(2) = -1.6984;
y_tout(3) = -2.554;
y_tout(4) = -2.3422;
y_tout(5) = -2.0212;

for count = 6:500
    y_tout(count) = b(1) + b(2)*y_tout(count-1) + b(3)*y_tout(count-2) + b(4)*y_tout(count-3)+ b(5)*y_tout(count-4)+ b(6)*y_tout(count-5)...
     + b(7)*x1_test(count-1) + b(8)*x1_test(count-2) + b(9)*x1_test(count-3)+ b(10)*x1_test(count-4)+ b(11)*x1_test(count-5)...
     + b(12)*x2_test(count-1) + b(13)*x2_test(count-2) + b(14)*x2_test(count-3)+ b(15)*x2_test(count-4)+ b(16)*x2_test(count-5)...
     + b(17)*x3_test(count-1) + b(18)*x3_test(count-2) + b(19)*x3_test(count-3)+ b(20)*x3_test(count-4)+ b(21)*x3_test(count-5)...
     + b(22)*x4_test(count-1) + b(23)*x4_test(count-2) + b(24)*x4_test(count-3)+ b(25)*x4_test(count-4)+ b(26)*x4_test(count-5);
end

b1 = b';
err = y_test1-y_tout;
T1 = 1:500;
figure;

plot(T1,err);

figure;
plot(T1,y_test1);
hold on
plot(T1,y_tout,'r');
hold off

sum1 = 0;
for count = 1:500
    sum1 = sum1 + (y_test1(count)-y_tout(count))*(y_test1(count)-y_tout(count));
end

RMSE = sqrt(sum1/500);

sum2 = 0;
for count = 1:500
    sum2 = sum2 + abs(y_test1(count)-y_tout(count));
end

MAE = sum2/500;


