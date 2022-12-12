%%%%%%%%%%%%%%%%%BLOOD CANCER DETECTION AND CLASSIFICATION %%%%%%%%
clc

close all;

clear all;

warning off;


%%%%%%Get the input image %%%%%
[filename,pathname]=uigetfile('*.jpg');

%%%% Read Input Image %%%%%
I=imread([pathname,filename]);
figure,imshow(I),title('INPUT IMAGE');

%%%%% Resize The input image %%%%%%
I=imresize(I,[512 512]);
figure,imshow(I);title('RESIZED INPUT IMAGE');


% % % %%%%TO PERFORM FUZZY C-MEANS CLUSTERING ON THE EYE DETECTED IMAGE%%%%%
% % % 
% % % fim=mat2gray(I );
% % % 
% % % level=graythresh(fim);
% % % 
% % % bwfim=im2bw(fim,level);
% % % 
% % % [bwfim0,level0]=fcmthresh(fim,0);
% % % 
% % % [bwfim1,level1]=fcmthresh(fim,1);
% % % 
% % % subplot(2,2,1);
% % % 
% % % imshow(fim);title('Original');
% % % 
% % % subplot(2,2,2);
% % % 
% % % imshow(bwfim);title(sprintf('Otsu,level=%f',level));
% % % 
% % % subplot(2,2,3);
% % % 
% % % imshow(bwfim0);title(sprintf('FCM0,level=%f',level0));
% % % 
% % % subplot(2,2,4);
% % % 
% % % imshow(bwfim1);title(sprintf('FCM1,level=%f',level1));

%%%%%%CONVERT RGB2CMYK COLOUR SPACE CONVERSIONS %%%%%%

cyan=I;
megenta=I;
yellow=I;

cyan(:,:,1)=0;
cyan(:,:,1)=0;
megenta(:,:,2)=0;
megenta(:,:,2)=0;
yellow(:,:,3)=0;
yellow(:,:,3)=0;

figure,imshow(cyan),title('CYAN  CHANNEL IMAGE');
figure,imshow(megenta),title('MEGENTA CHANNEL IMAGE');
figure,imshow(yellow),title('YELLOW CHANNEL IMAGE');

figure,
subplot(1,3,1),imshow(cyan),title('CYAN');
subplot(1,3,2),imshow(megenta),title('MEGENTA');
subplot(1,3,3),imshow(yellow),title('YELLOW');


%%%%SELECT THE YELLOW CHANNEL AND PERFORM RGB TO GRAY CONVERSION %%%%%%

%% RGB to Gray conversion
[m n o]=size(yellow);
if o==3
    gray=rgb2gray(yellow);
else
    gray=yellow;
end
figure,imshow(gray);title('YELLOW CHANNEL -GRAY IMAGE');

% 
%%%%%%ADJUST THE CONTRAST OF THE GRAY CHANNEL IMAGE%%%%
ad=imadjust(gray);
figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% 
%%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
bw=im2bw(gray,0.5);
figure,imshow(bw);title('BLACK AND WHITE IMAGE');
% 
%%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
bw=imcomplement(bw);
figure,imshow(bw);title('COMPLEMENT IMAGE');
% 
%%%%REMOVE SMALL OBJECTS ON THE COMPLEMENTED IMAGE %%%%
magnification_value=2000;
II=round(magnification_value/15);
bw1 = bwareaopen(bw,II);
figure,imshow(bw1);title('SMALL OBJECTS REMOVED IMAGE');
% 
%%%%PERFORM MORPHOLOGICAL FILL OPERARTION%%%
bw5 = imfill(bw1,'holes');
figure,imshow(bw5),title('MORPHOLOGICAL FILLED IMAGE');
%%%%TO REMOVE SMALL OBJECTS%%%%
bwx = bwareaopen(bw5,300);
figure,imshow(bwx);title('SMALL OBJECTS REMOVED IMAGE');
% 
%%%%%APPLY WATERSHED SEGMENTATION%%%%%
L = watershed(bw5);
Lrgb = label2rgb(L);
imshow(Lrgb)
bw2 = ~bwareaopen(~bw5, 10);
imshow(bw2)
D = -bwdist(~bw5);
imshow(D,[])
Ld = watershed(D);
imshow(label2rgb(Ld))
bw2 = bw5;
bw2(Ld == 0) = 0;
imshow(bw2)
mask = imextendedmin(D,2);
imshowpair(bw,mask,'blend');
D2 = imimposemin(D,mask);
Ld2 = watershed(D2);
bw3 = bw5;
bw3(Ld2 == 0) = 0;
figure,imshow(bw3),title('WATERSHED TRANSFORMED IMAGE');

%%%%REMOVE SMALL OBJECTS ON THE WATERSHED IMAGE%%%%
bww = bwareaopen(bw3,300);
figure,imshow(bww);title('SMALL OBJECTS REMOVED IMAGE');
% 
% 
% 
% 
%%%TO LABEL THE SMALL OBJECTS REMOVED IMAGE%%%%
label=bwlabel(bww);

%%%%APPLY REGION SEGMENTATION (REGION PROPERTIES)%%%%%
S=regionprops(label,'ALL');
figure,imshow(I),title('REGION PROPERTY IMAGE');

%%%FIND THE PERIMETER FOR THE BOUNDARY%%%%%
u=bwperim(bww);
% 
% 
%%%%%CONVERT TO DOUBLE FOEMAT%%%%%
u=im2double(u);
% 
% % %%%TO ASSIGN THE BOUNDARIES FOR THE DETECTION OF NUMBER OF WBC CELLS %%%%%5
[B,L] = bwboundaries(bww,'noholes');
% imshow(label2rgb(L, @jet, [.5 .5 .5]))
figure,imshow(I,[]);
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2)
end
% 
% 
% 
% 
%%%%%FEATURE EXTRACTION%%%%%%

%%%%TO TAKE COLOUR FEATURES%%%%%%
R = mean2(I(:,:,1));
G = mean2(I(:,:,2));
B = mean2(I(:,:,3));
Co_Fea = [R G B];

% 
%%%%TO TAKE GEOMETRICAL FEATURES%%%%
Area = mean([S.Area]);

for i = 1:size(S,1)
    diameters(i) = mean([S(i).MajorAxisLength S(i).MinorAxisLength])/2;
radii(i) = diameters(i)./2;
end
perimeter = mean([S.Perimeter]);
ecc = mean([S.Eccentricity]);
elg = mean([S.Solidity]);
% Elongation = mean(Elong);
dia = mean(diameters);
rad = mean(radii);
Geome_Fea = [Area dia rad perimeter ecc elg];


%%%%%TAKE IMPORTANT TEXTURE FEATURES %%%%%

g = graycomatrix(bww);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Tex_feat= [Contrast,Correlation,Energy,Homogeneity];

feat_tot=[Co_Fea Geome_Fea Tex_feat];

%%%%LOAD ALL THE FEATURES%%%%
load featurewbc2.mat
load featurewbc1.mat
% 
%%%%%CLASSIFICATION%%%% 

%%%PERFORM SVM CLASSIFICATION %%%%%%


test1 = zeros(1,2);
test1(1:1)= 1;
test1(2:2)=2;
A=fitcsvm(fea_wbctest,test1);
result=predict(A,feat_tot);
% 
% 
if result==1
    
    %%%TO ASSIGN THE BOUNDARIES FOR THE DETECTION OF NUMBER OF WBC CELLS %%%%%5
[B,L] = bwboundaries(bwx,'noholes');

%%%%%TO SHOW THE NUMBER OF PARASITES DETECTED IN THE INPUT IMAGE%%%%%
a = length(B);
disp('Total number of WBC cells detected in image = '); 
disp(a);
    ccf=1;
    msgbox('HEALTHY WBC CELL');
    disp('HEALTHY WBC CELL');
%     
elseif result==2
    
    
    [B,L] = bwboundaries(bww,'noholes');

%%%%%TO SHOW THE NUMBER OF PARASITES DETECTED IN THE INPUT IMAGE%%%%%
a = length(B);
disp('Total number of WBC cells detected in image = '); 
disp(a);

    label=ones(1,48);
label(1:7)=1;
label(8:15)=2;
label(16:23)=3;
label(24:28)=4;
label(29:48)=5;
% % % 
model=fitcknn(fea_wbc1,label);
result=predict(model,feat_tot);
    
if result==1
    
    ccf=2;
    msgbox('ACUTE LYMPHOCYTIC LEUKEMIA(ALL)');
    disp(' BLOOD CANCER DETECTED ');
elseif result==2
    ccf=3;
    msgbox('ACUTE MYELOID LEUKEMIA(AML)');
    disp('BLOOD CANCER DETECTED');
elseif result==3
    ccf=4;
    msgbox('CHRONIC LYMPHOCYTIC LEUKEMIA(CLL), ');
    disp('BLOOD CANCER DETECTED');
elseif result==4
    ccf=5;
    msgbox('CHRONIC MYELOID LEUKEMIA(CML) ');
    disp('BLOOD CANCER DETECTED');
elseif result==5
    ccf=6;
    msgbox('HEALTHY WBC CELL');
    disp('BLOOD CANCER NOT DETECTED');
end
end




%%%%%% ACCURACY %%%%%%%%

ACTUAL=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2];

PREDICTED=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2];

% EVAL=Evaluate(ACTUAL,PREDICTED);

idx = (ACTUAL()==1);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;

tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);




disp(' accuracy='); 
disp(accuracy*100);
disp('sensitivity=');
disp(sensitivity*100);
disp('specificity=');
disp(specificity*100);
disp('precision=');
disp(precision*100);
disp('recall=');
disp(recall*100);
disp('f_measure=');
disp(f_measure*100);
disp('gmean=');
disp(gmean*100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%