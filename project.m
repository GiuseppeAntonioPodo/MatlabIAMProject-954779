clear
clc
windowLenght=0.3;
disp('The program will start soon...')
stepLenght=0.09;
nfft=1024;
disp('I am setting the warnings to "off"')
warning('off')

% path genTrallRock
addpath(genpath(pwd))
disp('creating path for sounds files.... ')
TrRock=dir([pwd,'/rock/train/*.wav']);
TestRock=dir([pwd,'/rock/test/*.wav']);
TrSynth=dir([pwd,'/synth/train/*.wav']);
TestSynth=dir([pwd,'/synth/test/*.wav']);
TrPop=dir([pwd,'/pop/train/*.wav']);
TestPop=dir([pwd,'/pop/test/*.wav']);
disp('-----------------------------')
fprintf('\n')

% One featuring extraction group
disp('extracting MFFCs and chroma for the train Rock music set.....')
[TrChromaRock, TrMRock, TrallRock]=extracts(TrRock,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for train Pop music set.....')
[TrChromaPop, TrMPop, TrallPop]=extracts(TrPop,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for train Synth music set.....')
[TrChromaSynth, TrMSynth, TrallSynth]=extracts(TrSynth,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for test Rock music set.....')
[TestChromaRock, TestMRock, TestallRock]=extracts(TestRock,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for test Pop music set.....')
[TestChromaPop, TestMPop, TestallPop]=extracts(TestPop,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for test Synth music set.....')
[TestChromaSynth, TestMSynth, TestallSynth]=extracts(TestSynth,windowLenght,stepLenght);

% Grouping
disp('grouping the features.....')
fprintf('\n')
TrCepsAll=[TrMRock TrMSynth TrMPop];
TrChrAll=[TrChromaRock TrChromaSynth TrChromaPop];
TrAll=[TrallRock TrallSynth TrallPop];
TestCepsAll=[TestMRock TestMSynth TestMPop];
TestChrAll=[TestChromaRock TestChromaSynth TestChromaPop];
TestAll=[TestallRock TestallSynth TestallPop];

% Normalizing
disp('normalizing features.....')
fprintf('\n')
TrNCepsAll=normalized(TrCepsAll);
q=transpose(TrNCepsAll);
TrNChromaAll=transpose(normalized(TrChrAll));
TrNAll=transpose(normalized(TrAll));
TestNCepsAll=normalized(TestCepsAll);
t=transpose(TestNCepsAll);
TestNChromaAll=transpose(normalized(TestChrAll));
TestNAll=transpose(normalized(TestAll));

disp('-----------------------------')

% Labelling
disp('labelling the features of clean files.....')
AllTrCeps=cr_label(TrMRock, TrMSynth, TrMPop);
AllTrChroma=cr_label(TrChromaRock, TrChromaSynth, TrChromaPop);
TrlabAll=cr_label(TrallRock, TrallSynth, TrallPop);
GTceps=cr_label(TestMRock, TestMSynth, TestMPop);
GTchroma=cr_label(TestChromaRock, TestChromaSynth,TestChromaPop);
GTall=cr_label(TestallRock, TestallSynth, TestallPop);
k=[2, 8, 20, 35, 67, 76, 100];

disp('-----------------------------')

% KNN
fprintf('\n')
disp('computing KNN for Ceps...')

[rateCeps, predictCeps]=knn(k, q, AllTrCeps, t, GTceps);

fprintf('\n')
disp('computing knn for Chroma...')
[rateChroma, predictChroma]= knn(k,TrNChromaAll, AllTrChroma,TestNChromaAll,GTchroma );

fprintf('\n')
disp('computing knn for all...')
[rateAll, predictAll]= knn(k,TrNAll, TrlabAll,TestNAll,GTall );
fprintf('\n')
disp('-----------------------------')
%Noise and Denoise
fprintf('\n')
add_noise([pwd, '/pop/test/'], 'wav', [pwd, '/noise/'], 'wav', [pwd, '/noisy pop/'], [pwd, '/enhanced pop/'])
disp('-----------------------------')
fprintf('\n')
add_noise([pwd, '/rock/test/'], 'wav', [pwd, '/noise/'], 'wav', [pwd, '/noisy rock/'], [pwd, '/enhanced rock/'])
disp('-----------------------------')
fprintf('\n')
add_noise([pwd, '/synth/test/'], 'wav', [pwd, '/noise/'], 'wav', [pwd, '/noisy synth/'], [pwd, '/enhanced synth/'])
disp('-----------------------------')
fprintf('\n')
% Path creating 2
disp('creating path for noisy and enhanced files.... ')
NoiRo=dir([pwd,'/noisy rock/*.wav']);
EnaRo=dir([pwd,'/enhanced rock/*.wav']);
NoiSy=dir([pwd,'/noisy synth/*.wav']);
EnaSy=dir([pwd,'/enhanced synth/*.wav']);
NoiPop=dir([pwd,'/noisy pop/*.wav']);
EnaPop=dir([pwd,'/enhanced pop/*.wav']);


% Features extractiong 2
disp('extracting MFFCs and chroma for noisy Rock music set.....')
fprintf('\n')
[NoiChromeRo, NoiMRo, NoiAllRo]=extracts(NoiRo,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for enhanced Rock music set.....')
fprintf('\n')
[EnaChromaRo, EnaMRo, EnaAllRo]=extracts(EnaRo,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for noisy Pop music set.....')
fprintf('\n')
[NoiChromaPop, NoiMPop, NoiAllPop]=extracts(NoiPop,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for enhanced Pop music set.....')
fprintf('\n')
[EnaChromaPop, EnaMPop, EnaAllPop]=extracts(EnaPop,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for noisy Acoustic music set.....')
fprintf('\n')
[NoiChromaSy, NoiMSy, NoiAllSy]=extracts(NoiSy,windowLenght,stepLenght);
disp('extracting MFFCs and chroma for enhanced Acoustic music set.....')
fprintf('\n')
[EnaChromaSy, EnaMSy, EnaAllSy]=extracts(EnaSy,windowLenght,stepLenght);
fprintf('\n')

% Second grouping 
disp('grouping features of noisy and enhanced files...')
fprintf('\n')
AllNoCh=[NoiChromeRo NoiChromaPop NoiChromaSy];
AllNoM=[NoiMRo NoiMPop NoiMSy];
AllNo=[NoiAllRo NoiAllPop NoiAllSy];

AllEnCh=[EnaChromaRo EnaChromaPop EnaChromaSy];
AllEnM=[EnaMRo EnaMPop EnaMSy];
AllEn=[EnaAllRo EnaAllPop EnaAllSy];
fprintf('\n')

%normalizing
disp('normalizing for noisy and enhanced ')
NallNoCh=transpose(normalized(AllNoCh));
NallNoM=transpose(normalized(AllNoM));
NallNo=transpose(normalized (AllNo));
NallEnCh=transpose(normalized(AllEnCh));
NallEnM=transpose(normalized(AllEnM));
NallEn=transpose(normalized(AllEn));
fprintf('\n')
%labelling 2nd time
disp('labeling for noisy and enhanced....')
LabNoCh=cr_label(NoiChromeRo, NoiChromaPop, NoiChromaSy);
LabNoM=cr_label(NoiMRo, NoiMPop, NoiMSy);
LabNoAll=cr_label(NoiAllRo, NoiAllPop, NoiAllSy);

GtCh=cr_label(EnaChromaRo, EnaChromaPop, EnaChromaSy);
GtM=cr_label(EnaMRo, EnaMPop, EnaMSy);
EnLall=cr_label(EnaAllRo, EnaAllPop, EnaAllSy);

fprintf('\n')
disp('computing knn for noisy chroma')
[rateNoC, predictNoC]=knn(k,TrNChromaAll, AllTrChroma, NallNoCh ,LabNoCh );
fprintf('\n')
disp('computing knn for noisy ceps')
[rateNoM, predictNoM]=knn(k, q, AllTrCeps,NallNoM ,LabNoM);
fprintf('\n')
disp('computing knn for noisy All')
[rateNoAll, predictNoAll]=knn(k ,TrNAll, TrlabAll ,NallNo ,LabNoAll );
fprintf('\n')

disp('computing knn for enhanced chroma')
[rateEnC, predictEnC]=knn(k,TrNChromaAll, AllTrChroma, NallEnCh ,GtCh );
fprintf('\n')
disp('computing knn for enhanced ceps')
[rateEnM, predictEnM]=knn(k, q, AllTrCeps,NallEnM ,GtM);
fprintf('\n')
disp('computing knn for enhanced All')
[rateEnAll, predictEnAll]=knn(k ,TrNAll, TrlabAll ,NallEn ,EnLall);

%
figure
classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(GTchroma,predictChroma);
subplot(3,3,1)
confusionchart(c,classlabel,'title','confusion matrix chroma feats', 'normalization', 'row-normalized');

classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(GTceps,predictCeps);
subplot(3,3,2)
confusionchart(c,classlabel,'title','confusion matrix Ceps feats', 'normalization', 'row-normalized');

classlabel=["Rock"; "Synth"; "Pop"];
c=confusionmat(GTall,predictAll);
subplot(3,3,3)
confusionchart(c,classlabel,'title','confusion matrix chroma+ceps feats', 'normalization', 'row-normalized');

%
classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(LabNoCh,predictNoC);
subplot(3,3,4)
confusionchart(c,classlabel,'title','confusion matrix chroma feats noisy', 'normalization', 'row-normalized');

classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(LabNoM,predictNoM);
subplot(3,3,5)
confusionchart(c,classlabel,'title','confusion matrix Ceps feats noisy', 'normalization', 'row-normalized');

classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(LabNoAll,predictNoAll);
subplot(3,3,6)
confusionchart(c,classlabel,'title','confusion matrix chroma+ceps feats noisy', 'normalization', 'row-normalized');

%
classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(GtCh,predictEnC);
subplot(3,3,7)
confusionchart(c,classlabel,'title','confusion matrix chroma feats enhanced', 'normalization', 'row-normalized');

classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(GtM,predictEnM);
subplot(3,3,8)
confusionchart(c,classlabel,'title','confusion matrix Ceps feats enhanced', 'normalization', 'row-normalized');

classlabel=["Rock"; "Synth"; "Pop" ];
c=confusionmat(EnLall,predictEnAll);
subplot(3,3,9)
confusionchart(c,classlabel,'title','confusion matrix chroma+ceps feats enhanced', 'normalization', 'row-normalized');

