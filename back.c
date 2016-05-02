/******************************************************************************
Backpropagation  Luis Felipe Sant'Ana
*******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef int           BOOL;

#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL
#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1

#define sqr(x)        ((x)*(x))


typedef struct {                     	/* Uma Camada da Rede:                   */
        int           Units;         	/* - numero de unidades na camada		  */
        double*         Output;        	/* - saida da iesima unidade             */
        double*         Error;         	/* - erro da iesima unidade              */
        double**        Weight;        	/* - conecção dos pesos				  */
        double**        WeightSave;    	/* - pesos salvados					  */
        double**        dWeight;       	/* - delta para momentum   		      */
} LAYER;

typedef struct {                     	/* Uma REDE:                             */
        LAYER**       Layer;         	/* - camada da rede	                  	 */
        LAYER*        InputLayer;    	/* - entrada da Red                      */
        LAYER*        OutputLayer;   	/* - saida da camada                     */
        double          Alpha;         	/* - Momentum   	                     */
        double          Eta;           	/* - razão aprendizado                   */
        double          Gain;          	/* - Ganho da função sigmoid		     */
        double          Error;         	/* - erro total da rede                  */
} NET;


void InitializeRandoms()
{
  srand(4711);
}


int RandomEqualINT(int Low, int High)
{
  return rand() % (High-Low+1) + Low;
}      


double RandomEqualREAL(double Low, double High)
{
  return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}      


#define NUM_LAYERS    3
#define N             30
#define M             1
int                   Units[NUM_LAYERS] = {N, 10, M};

#define FIRST_YEAR    1
#define NUM_CARAC     20000

#define TRAIN_LWB     (N)
#define TRAIN_UPB     (179)
#define TRAIN_YEARS   (TRAIN_UPB - TRAIN_LWB + 1)
#define TEST_LWB      (180)
#define TEST_UPB      (259)
#define TEST_YEARS    (TEST_UPB - TEST_LWB + 1)
#define EVAL_LWB      (260)
#define EVAL_UPB      (NUM_CARAC - 1)
#define EVAL_YEARS    (EVAL_UPB - EVAL_LWB + 1)

double*                  Sunspots_;
double*                  Sunspots; 

double                  Mean;
double                  TrainError;
double                  TrainErrorPredictingMean;
double                  TestError;
double                  TestErrorPredictingMean;

FILE*                 f;

/* OK */
void ReadInput(){

	char url[]	=	"dataset.data";
	char *lettr = 	(char *) malloc(sizeof(char)*NUM_CARAC);
	int *xboxI 	= 	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *yboxI 	= 	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *widthI	= 	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *highI 	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *onpixI	= 	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *xbarI 	= 	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *ybarI	=   (int *) malloc(sizeof(int)*NUM_CARAC);
	int *x2barI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *y2barI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *xybarI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *x2ybrI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *xy2brI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *xegeI 	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *xegvyI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *yegeI 	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	int *yegvxI	=  	(int *) malloc(sizeof(int)*NUM_CARAC);
	FILE *arq;
	int i=0;
	
	
	arq = fopen(url, "r");
	if(arq == NULL){
	    printf("Erro, nao foi possivel abrir o arquivo\n");
	}else{
	    while(fscanf(arq,"%c,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", &lettr[i], &xboxI[i], &yboxI[i], &widthI[i], &highI[i], &onpixI[i], &xbarI[i], &ybarI[i], &x2barI[i], &y2barI[i], &xybarI[i], &x2ybrI[i], &xy2brI[i], &xegeI[i], &xegvyI[i], &yegeI[i], &yegvxI[i])!=EOF ){
			//printf("%c %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", lettr[i], xboxI[i], yboxI[i], widthI[i], highI[i], onpixI[i], xbarI[i], ybarI[i], x2barI[i], y2barI[i], xybarI[i], x2ybrI[i], xy2brI[i], xegeI[i], xegvyI[i], yegeI[i], yegvxI[i]);
			i++;
		}
	}
	fclose(arq);
		
	Sunspots = (double*) xboxI;
}

/* OK */
void NormalizeSunspots()
{
  int  Charc;
  double Min, Max;
	
  Min = MAX_REAL;
  Max = MIN_REAL;
  for (Charc=0; Charc<NUM_CARAC; Charc++) {
    Min = MIN(Min, Sunspots[Charc]);
    Max = MAX(Max, Sunspots[Charc]);
  }
  
  Mean = 0;
  
  Sunspots_ = (double*) malloc(sizeof(double)*NUM_CARAC);
  for (Charc=0; Charc<NUM_CARAC; Charc++) {
    Sunspots_[Charc] = 
    Sunspots [Charc] = ((Sunspots[Charc]-Min) / (Max-Min)) * (HI-LO) + LO;
    Mean += Sunspots[Charc] / NUM_CARAC;
  }
  
}

/* OK */
void InitializeApplication(NET* Net)
{
  int  Charc, i;
  double Out, Err;

  Net->Alpha = 0.5;
  Net->Eta   = 0.05;
  Net->Gain  = 1;
  ReadInput();		
  NormalizeSunspots();
 
  TrainErrorPredictingMean = 0;
  for (Charc=TRAIN_LWB; Charc<=TRAIN_UPB; Charc++) {
    for (i=0; i<M; i++) {
      Out = Sunspots[Charc+i];
      Err = Mean - Out;
      TrainErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  TestErrorPredictingMean = 0;
  for (Charc=TEST_LWB; Charc<=TEST_UPB; Charc++) {
    for (i=0; i<M; i++) {
      Out = Sunspots[Charc+i];
      Err = Mean - Out;
      TestErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  f = fopen("BPN.txt", "w");
  
}

/* OK */
void FinalizeApplication(NET* Net)
{
  fclose(f);
}

/* OK */
void GenerateNetwork(NET* Net)
{
  int l,i;

  Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));
   
  for (l=0; l<NUM_LAYERS; l++) {
    Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
      
    Net->Layer[l]->Units      = Units[l];
    Net->Layer[l]->Output     = (double*)  calloc(Units[l]+1, sizeof(double));
    Net->Layer[l]->Error      = (double*)  calloc(Units[l]+1, sizeof(double));
    Net->Layer[l]->Weight     = (double**) calloc(Units[l]+1, sizeof(double*));
    Net->Layer[l]->WeightSave = (double**) calloc(Units[l]+1, sizeof(double*));
    Net->Layer[l]->dWeight    = (double**) calloc(Units[l]+1, sizeof(double*));
    Net->Layer[l]->Output[0]  = BIAS;
      
    if (l != 0) {
      for (i=1; i<=Units[l]; i++) {
        Net->Layer[l]->Weight[i]     = (double*) calloc(Units[l-1]+1, sizeof(double));
        Net->Layer[l]->WeightSave[i] = (double*) calloc(Units[l-1]+1, sizeof(double));
        Net->Layer[l]->dWeight[i]    = (double*) calloc(Units[l-1]+1, sizeof(double));
      }
    }
  }
  Net->InputLayer  = Net->Layer[0];
  Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  Net->Alpha       = 0.9;
  Net->Eta         = 0.25;
  Net->Gain        = 1;
}

/* OK */
void RandomWeights(NET* Net)
{
  int l,i,j;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
      }
    }
  }
}

/* OK */
void SetInput(NET* Net, double* Input)
{
  int i;
   
  for (i=1; i<=Net->InputLayer->Units; i++) {
    Net->InputLayer->Output[i] = Input[i-1];
  }
}

/* OK */
void GetOutput(NET* Net, double* Output)
{
  int i;
   
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Output[i-1] = Net->OutputLayer->Output[i];
  }
}

/* OK */
void SaveWeights(NET* Net)
{
  int l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
      }
    }
  }
}

/* OK */
void RestoreWeights(NET* Net)
{
  int l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
      }
    }
  }
}

/* OK */
void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
  int  i,j;
  double Sum;

  for (i=1; i<=Upper->Units; i++) {
    Sum = 0;
    for (j=0; j<=Lower->Units; j++) {
      Sum += Upper->Weight[i][j] * Lower->Output[j];
    }
    Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
  }
}

/* OK */
void PropagateNet(NET* Net)
{
  int l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}

/* OK */
void ComputeOutputError(NET* Net, double* Target)
{
  int  i;
  double Out, Err;
   
  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Out = Net->OutputLayer->Output[i];
    Err = Target[i-1]-Out;
    Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;
    Net->Error += 0.5 * sqr(Err);
  }
}

/*OK*/
void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
  int  i,j;
  double Out, Err;
   
  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;
    for (j=1; j<=Upper->Units; j++) {
      Err += Upper->Weight[j][i] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
  }
}

/*OK*/
void BackpropagateNet(NET* Net)
{
  int l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}

/*OK*/
void AdjustWeights(NET* Net)
{
  int  l,i,j;
  double Out, Err, dWeight;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[i][j];
        Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
      }
    }
  }
}


/******************************************************************************
Simulação da REDE
******************************************************************************/
void SimulateNet(NET* Net, double* Input, double* Output, double* Target, BOOL Training)
{
  SetInput(Net, Input);
  PropagateNet(Net);
  GetOutput(Net, Output);
   
  ComputeOutputError(Net, Target);
  if (Training) {
    BackpropagateNet(Net);
    AdjustWeights(Net);
  }
}


void TrainNet(NET* Net, int Epochs)
{
  int  Charc, n;
  double Output[M];

  for (n=0; n<Epochs*TRAIN_YEARS; n++) {
    Charc = RandomEqualINT(TRAIN_LWB, TRAIN_UPB);
    SimulateNet(Net, &(Sunspots[Charc-N]), Output, &(Sunspots[Charc]), TRUE);
  }
}


void TestNet(NET* Net)
{
  int  Charc;
  double Output[M];

  TrainError = 0;
  for (Charc=TRAIN_LWB; Charc<=TRAIN_UPB; Charc++) {
    SimulateNet(Net, &(Sunspots[Charc-N]), Output, &(Sunspots[Charc]), FALSE);
    TrainError += Net->Error;
  }
  TestError = 0;
  for (Charc=TEST_LWB; Charc<=TEST_UPB; Charc++) {
    SimulateNet(Net, &(Sunspots[Charc-N]), Output, &(Sunspots[Charc]), FALSE);
    TestError += Net->Error;
  }
  fprintf(f, "\nNMSE is %0.3f no conjunto de treinamento e %0.3f no conjunto de teste",
             TrainError / TrainErrorPredictingMean,
             TestError / TestErrorPredictingMean);
}


void EvaluateNet(NET* Net)
{
  int  Charc;
  double Output [M];
  double Output_[M];

  fprintf(f, "\n\n\n");
  fprintf(f, "Iteração    Valor    Predição modo Open-Loop     Predição modo Closed-Loop\n");
  fprintf(f, "\n");
  for (Charc=EVAL_LWB; Charc<=EVAL_UPB; Charc++) {
    SimulateNet(Net, &(Sunspots [Charc-N]), Output,  &(Sunspots [Charc]), FALSE);
    SimulateNet(Net, &(Sunspots_[Charc-N]), Output_, &(Sunspots_[Charc]), FALSE);
    Sunspots_[Charc] = Output_[0];
    fprintf(f, "%d       %0.3f                   %0.3f                     %0.3f\n",
               FIRST_YEAR + Charc,
               Sunspots[Charc],
               Output [0],
               Output_[0]);
  }
}


int main()
{
  NET  Net;
  BOOL Stop;
  double MinTestError;

  InitializeRandoms();
  GenerateNetwork(&Net);
  RandomWeights(&Net);
  
  InitializeApplication(&Net);

  Stop = FALSE;
  MinTestError = MAX_REAL;
  do {
    TrainNet(&Net, 10);
    TestNet(&Net);
    if (TestError < MinTestError) {
      fprintf(f, " - salvando os pesos ...");
      MinTestError = TestError;
      SaveWeights(&Net);
    }
    else if (TestError > 1.2 * MinTestError) {
      fprintf(f, " - parando o treinamento e armazenando os pesos ...");
      Stop = TRUE;
      RestoreWeights(&Net);
    }
  } while (NOT Stop);

  TestNet(&Net);
  EvaluateNet(&Net);
   
  FinalizeApplication(&Net);
  
  return 0;
}