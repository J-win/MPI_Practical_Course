// Copyright ivanvikhrev
#include <assert.h>
#include <mpi.h>
#include <iostream>
#include "opencv2/opencv.hpp"

#define MainProc 0
#define ksize 11  // kernel for filter(and locality rows at the same time)

class img {
 public:
  cv::String imageName;
  cv::Mat pic;
  unsigned int cols;
  unsigned int rows;
  unsigned int  type;
  unsigned int  imgSize;
  img();
  explicit img(cv::Mat m);
  img(const img &m);
  ~img() {
  }
  bool loadImg(cv::String name);
  bool saveImg(cv::String name);
  void createImg(unsigned int height, unsigned int width);
  void show(cv::String wName);
};

img::img() {
  imageName = "";
  pic = cv::Mat();
  cols = 0;
  rows = 0;
  type = 0;
  imgSize = 0;
}

img::img(const img& m) {
  imageName = m.imageName;
  m.pic.copyTo(pic);
  cols = m.cols;
  rows = m.rows;
  type = m.type;
  imgSize = cols * rows;
}

img::img(cv::Mat m) {
  imageName = "";
  m.copyTo(pic);
  cols = m.cols;
  rows = m.rows;
  type = m.type();
  imgSize = cols * rows;
}

bool img::loadImg(cv::String name) {
  cv::String imageName = name;
  pic = imread(imageName, 0);
  if (pic.empty()) return false;
  cols = pic.cols;
  rows = pic.rows;
  imgSize = cols * rows;
  type = pic.type();
  return true;
}

bool img::saveImg(cv::String name) {
  return cv::imwrite(name, pic);
}

void img::createImg(unsigned int height, unsigned int width) {
  cv::RNG r;
  cols = width;
  rows = height;
  imgSize = cols*rows;
  type = CV_8U;
  pic = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
  r.fill(pic, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(255));
}

void applyFilter1(cv::Mat *img, int kernel_size) {
  cv::medianBlur(*img, *img, kernel_size);
}
void applyFilter2(cv::Mat *img, int kernel_size) {
  cv::blur(*img, *img, cv::Size(kernel_size, kernel_size));
}
void applyFilter3(cv::Mat *img, int kernel_size) {
  cv::GaussianBlur(*img, *img, cv::Size(kernel_size, kernel_size), 0);
}
void img::show(cv::String wName) {
  if (rows <= 1080 && cols <= 1920) {
    cv::imshow(wName, pic);
  }
}

const char* kAbout =
"parallel image smoothing sample."
"Size of image must be less then 10^9";

const char* kOptions =
"{ i image       | <none> | image to process           }"
"{ f filter      | 1      | filter(1,2 or 3)           }"
"{ r random      | true   |  generate random image     }"
"{ y height      | 600    |  image height(must be even)}"
"{ x width       | 600    |  image width(must be even) }"
"{ s show        | false  |  show image                }"
"{ v video       | false  |  web camera                }"
"{ h ? help usage|        |  print help message        }";

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, kOptions);
  parser.about(kAbout);

  int status = 0, rank = 0, size = 0;  // MPI vars
  img origImg, procImgS, procImgP;  // Original, sequence and parallel images
  cv::Mat frame;
  unsigned int cols = 0,
               rows = 0,
               image_size = 0,
               rows_to_one_proc = 0,
               remain_rows = 0,  // in case of eneven division
               remain_size = 0,  // in pixels
               portion = 0,  // pix to each process
               locality = 0,  // locality of filter
               filter = 1;
  int *scounts = NULL,  // use in scatterv and gatherv
      *displs = NULL;   // use in scatterv and gatherv
  cv::String imageName = "";
  uchar  *dataIN =  NULL,  // buffer
         *dataOUT = NULL,  // arr for result
         *p = NULL;  // pointer to data in gatherv
  double t1 = 0, t2 = 0;
  bool checkImg = true,
       exit = false;

  status = MPI_Init(&argc, &argv);
  if (status != MPI_SUCCESS) {
    return -1;
  }

  status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (status != MPI_SUCCESS) {
    return -1;
  }


  status = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (status != MPI_SUCCESS) {
    return -1;
  }

  if (parser.get<bool>("help")) {
    if (rank == MainProc )
      parser.printMessage();
    MPI_Finalize();
    return 0;
  }

  if (!parser.get<bool>("v")) {
      if (rank == MainProc) {
          if (parser.has("i")) {
              imageName = parser.get<cv::String>("i");
              checkImg = origImg.loadImg(imageName);
              std::cout << "Path to image: " << imageName << std::endl;;
              if (!checkImg) {
                  std::cout << "Cant load image" << std::endl;
                  MPI_Abort(MPI_COMM_WORLD, 1);
              }
          } else {
              if (parser.get<bool>("r")) {
                  std::cout << "Random Image" << std::endl;
                  rows = parser.get<unsigned int>("y");
                  std::cout << "Rows: " << rows << std::endl;
                  cols = parser.get<unsigned int>("x");
                  std::cout << "Cols: " << cols << std::endl;
                  origImg.createImg(rows, cols);
              }
          }
          if (parser.has("f")) {
              filter = parser.get<unsigned int>("f");
              if (filter != 1 && filter != 2 && filter != 3) {
                  std::cout << "filter`s value 1,2 or 3" << std::endl;
                  filter = 1;
              }
              std::cout << "filter: " << filter << std::endl;
          }

          cols = origImg.cols;
          rows = origImg.rows;
          image_size = origImg.imgSize;
          rows_to_one_proc = rows / size;
          remain_rows = rows - size * rows_to_one_proc;
          portion = rows_to_one_proc * cols;
          procImgS = img(origImg);
          t1 = MPI_Wtime();

          if (filter == 1) {
            applyFilter1(&procImgS.pic, ksize);
          } else {
              if (filter == 2) {
                applyFilter2(&procImgS.pic, ksize);
              } else {
                  applyFilter3(&procImgS.pic, ksize);
              }
          }
          t2 = MPI_Wtime();
          std::cout << "Sequential Time: " << t2 - t1 << std::endl;
          dataIN = new uchar[portion + 2 * cols*ksize + remain_rows * cols];
          dataOUT = new uchar[image_size];
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t1 = MPI_Wtime();
      MPI_Bcast(&remain_rows, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
      MPI_Bcast(&rows_to_one_proc, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
      MPI_Bcast(&portion, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
      MPI_Bcast(&cols, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
      MPI_Bcast(&filter, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);

      locality = cols * ksize;
      remain_size = cols * remain_rows;
      if (rank != 0)
          dataIN = new uchar[portion + 2 * locality];
      displs = new int[size];
      scounts = new int[size];

      for (int i = 1; i < size; i++) {
          displs[i] = remain_size + i * portion - locality;
          scounts[i] = portion + 2 * locality;
      }
      displs[0] = 0;
      scounts[0] = portion + remain_size + locality;
      scounts[size - 1] = portion + locality;
      if (size == 1)
          scounts[0] = portion;
      MPI_Scatterv(origImg.pic.data, scounts, displs, MPI_UNSIGNED_CHAR, dataIN,
          scounts[rank], MPI_UNSIGNED_CHAR, MainProc, MPI_COMM_WORLD);

      cv::Mat tmp(scounts[rank] / cols, cols, CV_8U, dataIN);

      if (filter == 1) {
        applyFilter1(&tmp, ksize);
      } else {
          if (filter == 2) {
            applyFilter2(&tmp, ksize);
          } else {
              applyFilter3(&tmp, ksize);
          }
      }

      p = tmp.data + cols * ksize;  // pointer to data
      if (rank == MainProc) {
          p = tmp.data;
      }
      for (int i = 1; i < size; i++) {
          displs[i] = i * portion + remain_size;;
          scounts[i] = portion;
      }
      displs[0] = 0;
      scounts[0] = portion + remain_size;
      MPI_Gatherv(p, scounts[rank], MPI_UNSIGNED_CHAR, dataOUT, scounts,
          displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      t2 = MPI_Wtime();
      if (rank == MainProc) {
          procImgP = img(cv::Mat(origImg.rows, origImg.cols, CV_8U, dataOUT));
          std::cout << "Parallel Time:   " << t2 - t1 << std::endl;

          if (parser.get<bool>("s")) {
              origImg.show(static_cast<cv::String>("Original"));
              procImgS.show(static_cast<cv::String>("Sequence processed"));
              procImgP.show(static_cast<cv::String>("Parallel processed"));
              cv::waitKey(0);
          }
          if (!origImg.saveImg("Original.png")) {
              std::cout << " Can`t save image 0 " << std::endl;
          }
          if (!procImgS.saveImg("smooth_image_seq.png")) {
              std::cout << " Can`t save image 1" << std::endl;
          }
          if (!procImgP.saveImg("smooth_image_par.png")) {
              std::cout << " Can`t save image 2" << std::endl;
          }
          unsigned int count = 0;
          for (unsigned int i = 0; i < image_size; i++) {
              if (procImgS.pic.data[i] != procImgP.pic.data[i]) {
                  count++;
              }
          }
          std::cout << "Different values: " << count << std::endl;
      }
  } else {
      if (rank == MainProc) {
          if (parser.has("f")) {
              filter = parser.get<unsigned int>("f");
              if (filter != 1 && filter != 2 && filter != 3) {
                  std::cout << "filter`s value 1,2 or 3" << std::endl;
                  filter = 1;
              }
              std::cout << "filter: " << filter << std::endl;
          }
          cv::VideoCapture capture;
          capture.open(0);
          if (!capture.isOpened()) {
            std::cout << "Can`t open webcam" << std::endl;
            exit = true;
            MPI_Finalize();
            return 0;
          }
          capture.read(frame);
          origImg = img(frame);
          cv::cvtColor(origImg.pic, origImg.pic, cv::COLOR_BGR2GRAY);
          cols = origImg.cols;
          rows = origImg.rows;
          image_size = origImg.imgSize;
          rows_to_one_proc = rows / size;
          remain_rows = rows - size * rows_to_one_proc;
          portion = rows_to_one_proc * cols;
          locality = cols * ksize;
          remain_size = cols * remain_rows;
          displs = new int[size];
          scounts = new int[size];
          dataIN = new uchar[portion + 2 * cols*ksize + remain_rows * cols];
          dataOUT = new uchar[image_size];
          MPI_Bcast(&remain_rows, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          MPI_Bcast(&rows_to_one_proc, 1, MPI_UNSIGNED, MainProc,
            MPI_COMM_WORLD);
          MPI_Bcast(&portion, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          MPI_Bcast(&cols, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          MPI_Bcast(&filter, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          char c = 0;
          while (1) {
            capture >> frame;
            origImg = img(frame);
            cv::cvtColor(origImg.pic, origImg.pic, cv::COLOR_BGR2GRAY);
            for (int i = 1; i < size; i++) {
              displs[i] = remain_size + i * portion - locality;
              scounts[i] = portion + 2 * locality;
            }
            displs[0] = 0;
            scounts[0] = portion + remain_size + locality;
            scounts[size - 1] = portion + locality;
            if (size == 1)
              scounts[0] = portion;

            MPI_Scatterv(origImg.pic.data, scounts, displs, MPI_UNSIGNED_CHAR,
             dataIN, scounts[rank], MPI_UNSIGNED_CHAR, MainProc,
               MPI_COMM_WORLD);

            cv::Mat tmp(scounts[rank] / cols, cols, CV_8U, dataIN);
            if (filter == 1) {
              applyFilter1(&tmp, ksize);
            } else {
                if (filter == 2) {
                  applyFilter2(&tmp, ksize);
                } else {
                  applyFilter3(&tmp, ksize);
                }
            }
            p = tmp.data;
            for (int i = 1; i < size; i++) {
              displs[i] = i * portion + remain_size;;
              scounts[i] = portion;
            }
            displs[0] = 0;
            scounts[0] = portion + remain_size;

            MPI_Gatherv(p, scounts[rank], MPI_UNSIGNED_CHAR, dataOUT, scounts,
              displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            procImgP = img(cv::Mat(origImg.rows, origImg.cols, CV_8U, dataOUT));
            cv::imshow("Processed", procImgP.pic);
            c = cv::waitKey(1);
            if (c == 27) exit = true;
            MPI_Bcast(&exit, 1, MPI_C_BOOL, MainProc, MPI_COMM_WORLD);
            if (c == 27) break;
          }
      } else {
          MPI_Bcast(&remain_rows, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          MPI_Bcast(&rows_to_one_proc, 1, MPI_UNSIGNED, MainProc,
            MPI_COMM_WORLD);
          MPI_Bcast(&portion, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          MPI_Bcast(&cols, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
          MPI_Bcast(&filter, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);

          locality = cols * ksize;
          remain_size = cols * remain_rows;
          dataIN = new uchar[portion + 2 * locality];
          displs = new int[size];
          scounts = new int[size];
          while (1) {
            if (exit == true) break;
            for (int i = 1; i < size; i++) {
              displs[i] = remain_size + i * portion - locality;
              scounts[i] = portion + 2 * locality;
            }
            displs[0] = 0;
            scounts[0] = portion + remain_size + locality;
            scounts[size - 1] = portion + locality;
            if (size == 1)
              scounts[0] = portion;

            MPI_Scatterv(origImg.pic.data, scounts, displs, MPI_UNSIGNED_CHAR,
              dataIN, scounts[rank], MPI_UNSIGNED_CHAR, MainProc,
                MPI_COMM_WORLD);

            cv::Mat tmp(scounts[rank] / cols, cols, CV_8U, dataIN);
            if (filter == 1) {
              applyFilter1(&tmp, ksize);
            } else {
                if (filter == 2) {
                  applyFilter2(&tmp, ksize);
                } else {
                    applyFilter3(&tmp, ksize);
                }
            }
            p = tmp.data + cols * ksize;  // pointer to data
            for (int i = 1; i < size; i++) {
              displs[i] = i * portion + remain_size;;
              scounts[i] = portion;
            }
            displs[0] = 0;
            scounts[0] = portion + remain_size;
            MPI_Gatherv(p, scounts[rank], MPI_UNSIGNED_CHAR, dataOUT, scounts,
                displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            MPI_Bcast(&exit, 1, MPI_C_BOOL, MainProc, MPI_COMM_WORLD);
          }
        }
  }
  delete[] dataIN;
  delete[] dataOUT;
  delete[] displs;
  delete[] scounts;
  status = MPI_Finalize();
  if (status != MPI_SUCCESS) {
    return -1;
  }
  return 0;
}
