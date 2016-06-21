function main()
tstart=tic;
SamNum=4995;              %ѵ�������� 
TestSamNum=495;            %���������� 
SP = 0.6; % ���ڵ���չ���� 
ErrorLimit = 0.9; % Ŀ����� 
% ����Ŀ�꺯���������������� 

N = importdata('C:\Users\Dazzal\Documents\Tencent Files\175760063\FileRecv\����ѧϰ����ҵ\�����ʱ������Ԥ��\train_X.txt');
Y1 = importdata('C:\Users\Dazzal\Documents\Tencent Files\175760063\FileRecv\����ѧϰ����ҵ\�����ʱ������Ԥ��\train_y.txt');
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
%��������
x_test = importdata('C:\Users\Dazzal\Desktop\������\����ѧϰ����ҵ\�����ʱ������Ԥ��\test_X.txt');
y_test = importdata('C:\Users\Dazzal\Desktop\������\����ѧϰ����ҵ\�����ʱ������Ԥ��\test_y.txt');
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



[InDim,MaxUnitNum] = size(SamIn); % ��������ά��������������ڵ��� 
% �������ڵ������ 
Distance = dist(SamIn',SamIn); 
HiddenUnitOut = radbas(Distance/SP); 
PosSelected = []; 
VectorsSelected = []; 
HiddenUnitOutSelected = []; 
ErrHistory = []; % ���ڼ�¼ÿ���������ڵ���ѵ����� 

VectorsSelectFrom = HiddenUnitOut; 

dd = sum((SamOut.*SamOut)')'; 
for k = 1 : MaxUnitNum 
% ��������ڵ����ʸ����Ŀ�����ʸ���ļн�ƽ��ֵ 
PP = sum(VectorsSelectFrom.*VectorsSelectFrom)'; 
Denominator = dd * PP'; 
[xxx,SelectedNum] = size(PosSelected); 
if SelectedNum>0, 
[lin,xxx] = size(Denominator); 
Denominator(:,PosSelected) = ones(lin,1); 
end 
Angle = ((SamOut*VectorsSelectFrom) .^ 2) ./ Denominator; 

% ѡ��������ͶӰ��ʸ�����õ���Ӧ���������� 
[value,pos] = max(Angle); 
PosSelected = [PosSelected pos]; 

% ����RBF ��ѵ����� 
HiddenUnitOutSelected = [HiddenUnitOutSelected; HiddenUnitOut(pos,:)]; 
HiddenUnitOutEx = [HiddenUnitOutSelected; ones(1,SamNum)]; 
W2Ex = SamOut*pinv(HiddenUnitOutEx); % �ù�������������Ȩֵ 
W2 = W2Ex(:,1:k); % �õ����Ȩֵ 
B2 = W2Ex(:,k+1); % �õ�ƫ�� 
NNOut = W2*HiddenUnitOutSelected+B2; % ����RBF ����� 
SSE = sumsqr(SamOut-NNOut) 

% ��¼ÿ���������ڵ���ѵ����� 
ErrHistory = [ErrHistory SSE]; 

if SSE < ErrorLimit, break, end 

% ��Gram-Schmidt ������ 
NewVector = VectorsSelectFrom(:,pos); 
ProjectionLen = NewVector' * VectorsSelectFrom / (NewVector'*NewVector); 
VectorsSelectFrom = VectorsSelectFrom - NewVector * ProjectionLen; 
end
 
UnitCenters = SamIn(:,PosSelected);

% ���� 
TestDistance = dist(UnitCenters',TestSamIn);
TestHiddenUnitOut = radbas(TestDistance/SP); 
TestNNOut = W2*TestHiddenUnitOut+B2; 
t=1:495;
TestSamOut1 = TestSamOut[6:500]; 
plot(t,TestNNOut,'k-')
grid
hold on
plot(t,TestSamOut1,'b+')


k 
UnitCenters 
W2 
B2 
tend=toc(tstart);
tend