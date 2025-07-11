#include "tnnr.h"

inline float traceNorm(Mat &X){
  Mat w;
  SVD::compute(X,w);
  return sum(w)[0];
}
// mask 8-bit 1 channel 0 and 255
// other matrices must be of CV_32FC1 type
Mat APGL(Mat &A, Mat &B, Mat &X, Mat &M, Mat &mask, float eps, float lambda){
  Mat AB = A.t() * B;
  Mat BA = B.t() * A;

  int W = X.cols;
  int H = X.rows;
  
  Mat lastX = Mat::zeros(H, W, CV_32FC1);
  Mat Y = Mat::zeros(H, W, CV_32FC1);
  X.copyTo(lastX);
  lastX.copyTo(Y);


  float tlast = 1.0;
  float t = 1.0;
  Mat XX = Mat::zeros(H, W, CV_32FC1);
  X.copyTo(XX);

  Mat known;
  mask.convertTo(known, CV_32FC1, 1.0/255.0);


  Mat u, sigma, v;

  float objval = 0.0;
  float objval_last = 0.0;
  int k = 1;

  for(k = 1; k < 201; k++){
    Mat tmp;
    multiply(Y-M, known, tmp);
    Mat G = Y + t * (AB - lambda * tmp);
    //svd for G
    SVD::compute(G, sigma, u, v);

    sigma = max(sigma - t, 0.0);
    sigma = Mat::diag(sigma);
    X = u*sigma*v;
    t = (1 + sqrt(1+4*tlast*tlast))/2;
    Y = X + (tlast-1)/(t)*(X - lastX);

    X.copyTo(XX);
    X.copyTo(lastX);
    tlast = t;

    multiply(X-M, known, tmp);
    float tr = trace(X*BA)[0];
    objval = traceNorm(X)  - tr + lambda / 2.0 * pow(norm(tmp,NORM_L2),2); 
    if(k >= 2 && -(objval - objval_last) < eps) break;

    objval_last = objval;
  }
  //  cout << "APGL total Iterations = " << k << endl;
  return XX;
}


Mat TNNR(Mat &im0, Mat &mask, int lower_R, int upper_R, float lambda){
  Mat X, M, A, B;

  im0.convertTo(X, CV_32FC1);
  X.copyTo(M);

  int W = X.cols;
  int H = X.rows;
  Mat X_rec = Mat::zeros(H, W, CV_32FC1);
  Mat X_rec_last = Mat::zeros(H, W, CV_32FC1);
  float eps = 0.1;
  int number_out_of_iter = 10;

  
  for(int R = lower_R; R <= upper_R; R++){
    for(int out_iter = 1; out_iter < 11; ++out_iter){
      //	  cout << "  TNNR  iter = " << out_iter << endl;
      Mat u,sigma,v;
      SVD::compute(X, sigma, u, v);
      A = u(Range::all(), Range(0, R - 1));
      A = A.t();
      B = v(Range(0, R - 1), Range::all());

      X_rec = APGL(A, B, X, M, mask, eps, lambda);

      if(out_iter >= 2 && norm(X_rec - X_rec_last) / norm(M,NORM_L2) < .01){
        X_rec.copyTo(X);
        break;
      }

      X_rec.copyTo(X);
      X_rec.copyTo(X_rec_last);	     
	  }
  } 

  return X;
}

Mat TNNR_APGL(Mat& im0, Mat& mask, float SVD_ratio, float lambda, float eps) {  
  int H = im0.rows;
  int W = im0.cols;
  int R = (int)(min(H, W) * SVD_ratio);
  cout<< "R = " << R << endl;

  Mat A, Sigma, B;
  SVD::compute(im0, Sigma, A, B);
  A = A(Range::all(), Range(0, R));
  B = B(Range(0, R), Range::all());
  
  Mat AB;
  gemm(A, B, 1.0, Mat(), 0.0, AB);

  Mat X, Xlast, Y;
  im0.copyTo(X);
  im0.copyTo(Y);
  float t = 1.0, last_t, obj_val;

  float input_norm = norm(im0.mul(mask), NORM_L2);

  float sigma_r_init = Sigma.at<float>(R - 1, 0);

  for(int i=0; i<200; i++){
    // New X
    X.copyTo(Xlast);
    Mat tmp = Y + t * (AB - lambda * ((Y-im0).mul(mask)));
    Mat u, s, v;
    SVD::compute(tmp, s, u, v);

    s = max(s - t, 0.0);
    s = Mat::diag(s);
    X = u * s * v;

    // New t
    last_t = t;
    t = (1 + sqrt(1 + 4 * last_t * last_t)) / 2.0; // update t

    // New Y
    Y = X + ((last_t - 1) / t) * (X - Xlast);

    /*
    // Objective
    float fro_norm = norm((X - im0).mul(mask), NORM_L2);
    fro_norm = fro_norm * fro_norm; // square the norm
    obj_val = traceNorm(X) - trace(X * AB.t())[0] + lambda / 2.0 * fro_norm; // objective function value

    // Change in X
    Mat diff = (X - Xlast).mul(mask);
    float norm_diff = norm(diff, NORM_L2);
    cout << "Iteration " << i << ", norm_diff = " << norm_diff << " obj_val = " << obj_val << endl;

    cout<< norm_diff << " " << input_norm << " " << eps << endl;
    */
  }
  
  return X;
}

Mat TNNR_ADMM(Mat& im0, Mat& mask, float SVD_ratio, float beta, float eps){
  int Height = im0.rows;
  int Width = im0.cols;
  int R = (int)(min(Height, Width) * SVD_ratio);
  cout<< "R = " << R << endl;

  Mat A, Sigma, B;
  SVD::compute(im0, Sigma, A, B);
  A = A(Range::all(), Range(0, R));
  B = B(Range(0, R), Range::all());
  
  Mat AB;
  gemm(A, B, 1.0, Mat(), 0.0, AB);

  Mat X, W, Y;
  im0.copyTo(X);
  im0.copyTo(W);
  im0.copyTo(Y);

  float input_norm = norm(im0.mul(mask), NORM_L2);

  float sigma_r_init = Sigma.at<float>(R - 1, 0);

  for(int i=0; i<200; i++){
    Mat tmp = W - Y/beta;
    Mat u, s, v;
    SVD::compute(tmp, s, u, v);

    s = max(s - 1.0/beta, 0.0);
    s = Mat::diag(s);
    X = u * s * v;

    W = X + (AB + Y)/beta;
    W = W.mul(1-mask) + im0.mul(mask); // apply mask

    Y = Y + beta * (X - W);

    //Mat diff = (Xnew - X).mul(mask);
    //cout<< "Iteration " << i << " norm_diff = " << norm(diff, NORM_L2) << endl;
  }

  return X;
}