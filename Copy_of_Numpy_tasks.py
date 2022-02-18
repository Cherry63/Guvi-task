{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Cherry63/Guvi-task/blob/main/Copy_of_Numpy_tasks.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aT_7K9xdqSeG"
      },
      "source": [
        "# Numpy\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "amNCJTo8qSeJ"
      },
      "source": [
        "#### 1. Import the numpy package under the name `np` (★☆☆)\n",
        "\n",
        "> Indented block\n",
        "\n",
        "\n",
        "(**hint**: import … as …)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 425,
      "metadata": {
        "collapsed": true,
        "id": "Q14IUPV8qSeK"
      },
      "outputs": [],
      "source": [
        "import numpy as np"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6Pb_nN4FqSeM"
      },
      "source": [
        "#### 2. Print the numpy version and the configuration (★☆☆) \n",
        "(**hint**: np.\\_\\_version\\_\\_, np.show\\_config)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 426,
      "metadata": {
        "id": "ycfMwoZWqSeN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c0fcce8e-3f0e-4949-ffc8-634dc888ed99"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1.21.5\n",
            "blas_mkl_info:\n",
            "  NOT AVAILABLE\n",
            "blis_info:\n",
            "  NOT AVAILABLE\n",
            "openblas_info:\n",
            "    libraries = ['openblas', 'openblas']\n",
            "    library_dirs = ['/usr/local/lib']\n",
            "    language = c\n",
            "    define_macros = [('HAVE_CBLAS', None)]\n",
            "    runtime_library_dirs = ['/usr/local/lib']\n",
            "blas_opt_info:\n",
            "    libraries = ['openblas', 'openblas']\n",
            "    library_dirs = ['/usr/local/lib']\n",
            "    language = c\n",
            "    define_macros = [('HAVE_CBLAS', None)]\n",
            "    runtime_library_dirs = ['/usr/local/lib']\n",
            "lapack_mkl_info:\n",
            "  NOT AVAILABLE\n",
            "openblas_lapack_info:\n",
            "    libraries = ['openblas', 'openblas']\n",
            "    library_dirs = ['/usr/local/lib']\n",
            "    language = c\n",
            "    define_macros = [('HAVE_CBLAS', None)]\n",
            "    runtime_library_dirs = ['/usr/local/lib']\n",
            "lapack_opt_info:\n",
            "    libraries = ['openblas', 'openblas']\n",
            "    library_dirs = ['/usr/local/lib']\n",
            "    language = c\n",
            "    define_macros = [('HAVE_CBLAS', None)]\n",
            "    runtime_library_dirs = ['/usr/local/lib']\n",
            "Supported SIMD extensions in this NumPy install:\n",
            "    baseline = SSE,SSE2,SSE3\n",
            "    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2\n",
            "    not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL\n",
            "None\n"
          ]
        }
      ],
      "source": [
        "\n",
        "print(np.__version__)\n",
        "print(np.show_config())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7ucQdHkTqSeN"
      },
      "source": [
        "#### 3. Create a null vector of size 10 (★☆☆) \n",
        "(**hint**: np.zeros)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 427,
      "metadata": {
        "id": "hiVfmuJ6qSeO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9a2dd5be-b15d-4dd5-92d9-f12048e3d329"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "def null_array(n):\n",
        "    x=np.zeros(n)\n",
        "    return x\n",
        "print(null_array(10))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fV-u5GluqSeP"
      },
      "source": [
        "#### 4.  How to find the memory size of any array (★☆☆) \n",
        "(**hint**: size, itemsize)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 428,
      "metadata": {
        "id": "6Uyv-oRcqSeQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cc252689-146e-4f3f-940e-634dfa3a456e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Size of the array:  3\n",
            "Memory size of one array element in bytes:  8\n"
          ]
        }
      ],
      "source": [
        "\n",
        "import numpy as np\n",
        "\n",
        "P = np.array([100,20,34])\n",
        "print(\"Size of the array: \",P.size)\n",
        "print(\"Memory size of one array element in bytes: \",P.itemsize)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tHFpxluJqSeR"
      },
      "source": [
        "#### 5.  How to get the documentation of the numpy add function from the \n",
        "\n",
        "---\n",
        "\n",
        "command line? (★☆☆) \n",
        "(**hint**: np.info)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 429,
      "metadata": {
        "collapsed": true,
        "id": "NQe7jyPIqSeS"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "np.info(np.add)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EiLRahLoqSeT"
      },
      "source": [
        "#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) \n",
        "(**hint**: array\\[4\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 430,
      "metadata": {
        "id": "CIeZRDfuqSeU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "aeef15c9-ac82-4fa3-a8b1-5fd65e45e19c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "X = np.zeros(10)\n",
        "X[4] = 1\n",
        "print(X)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VhdJBFQIqSeU"
      },
      "source": [
        "#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) \n",
        "(**hint**: np.arange)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 431,
      "metadata": {
        "id": "7kOH8AbYqSeV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "07fe2772-80d4-4f79-86fb-9fd94efc3274"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33\n",
            " 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "D = np.arange(10,50)\n",
        "print(D)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bm-PrqLuqSeW"
      },
      "source": [
        "#### 8.  Reverse a vector (first element becomes last) (★☆☆) \n",
        "(**hint**: array\\[::-1\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 432,
      "metadata": {
        "id": "-i9A_d5cqSeW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "62e85261-9b85-421b-82a9-c446b816701d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26\n",
            " 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10]\n"
          ]
        }
      ],
      "source": [
        "Z = np.arange(10,50)\n",
        "Z= Z[::-1]\n",
        "print (Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hi0OatWoqSeX"
      },
      "source": [
        "#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) \n",
        "(**hint**: reshape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 433,
      "metadata": {
        "id": "802xEntCqSeX",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "34715713-fdd7-459c-82ee-678ad2f07418"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 1 2]\n",
            " [3 4 5]\n",
            " [6 7 8]]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "O =  np.arange(0,9).reshape(3,3)\n",
        "print(O)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PhJtEmD7qSeY"
      },
      "source": [
        "#### 10. Find indices of non-zero elements from \\[1,2,0,0,4,0\\] (★☆☆) \n",
        "(**hint**: np.nonzero)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 434,
      "metadata": {
        "id": "1ZzdDMoHqSeY",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "be67fd22-0c23-44a8-e51a-e6fab475a1a0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(array([0, 1, 4]),)\n"
          ]
        }
      ],
      "source": [
        "I = np.nonzero([1,2,0,0,4,0])\n",
        "print(I)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XNYONM9OqSeZ"
      },
      "source": [
        "#### 11. Create a 3x3 identity matrix (★☆☆) \n",
        "(**hint**: np.eye)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 435,
      "metadata": {
        "id": "vQKW4H3wqSea",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "355b0473-c4e9-4e9a-a4a0-8f508cd483c0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1. 0. 0.]\n",
            " [0. 1. 0.]\n",
            " [0. 0. 1.]]\n"
          ]
        }
      ],
      "source": [
        "cehrry= np.eye(3)\n",
        "print (cehrry)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "un0HeXOCqSea"
      },
      "source": [
        "#### 12. Create a 3x3x3 array with random values (★☆☆) \n",
        "(**hint**: np.random.random)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 436,
      "metadata": {
        "collapsed": true,
        "id": "NStTkg9XqSea",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d3b0b10f-09b5-4af0-b1fc-d9704bc01d07"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[[0.83438194 0.03131294 0.82662688]\n",
            "  [0.25480111 0.65120632 0.97093947]\n",
            "  [0.12262082 0.16349402 0.65033749]]\n",
            "\n",
            " [[0.36007299 0.35279972 0.41724412]\n",
            "  [0.87715661 0.70529477 0.33172333]\n",
            "  [0.27634584 0.53294349 0.90526955]]\n",
            "\n",
            " [[0.49680285 0.72053678 0.93801944]\n",
            "  [0.89126517 0.58276167 0.77774657]\n",
            "  [0.98391136 0.52594325 0.57787251]]]\n"
          ]
        }
      ],
      "source": [
        "i = np.random.random((3,3,3))\n",
        "print (i)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RPaZst-DqSeb"
      },
      "source": [
        "#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) \n",
        "(**hint**: min, max)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 437,
      "metadata": {
        "id": "L9F0c9OmqSeb",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ae7e0456-0f9e-4cd5-e6e1-d4dc28c1ac63"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.019614043795864733 0.9695954469469426\n"
          ]
        }
      ],
      "source": [
        "A = np.random.random((10,10))\n",
        "Amin, Amax = A.min(), A.max()\n",
        "print(Amin, Amax)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DEHVly65qSeb"
      },
      "source": [
        "#### 14. Create a random vector of size 30 and find the mean value (★☆☆) \n",
        "(**hint**: mean)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 438,
      "metadata": {
        "id": "9J3hV7YuqSec",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ff29bbb8-bff6-4d5b-e4da-5209fb0d25ac"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.5144305226178396\n"
          ]
        }
      ],
      "source": [
        "O = np.random.random(30)\n",
        "F = O.mean()\n",
        "print (F)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r1s2OvcDqSec"
      },
      "source": [
        "#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) \n",
        "(**hint**: array\\[1:-1, 1:-1\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 439,
      "metadata": {
        "id": "7rnnLWjGqSec"
      },
      "outputs": [],
      "source": [
        "a = np.ones((10,10))\n",
        "a[1:-1, 1:-1]=0\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_Nblz5DQqSec"
      },
      "source": [
        "#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) \n",
        "\n",
        "*   List item\n",
        "*   List item\n",
        "\n",
        "\n",
        "(**hint**: np.pad)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 440,
      "metadata": {
        "id": "dqaY_ZGIqSed",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ab067041-b83e-4c69-b6bc-76d267509187"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 1. 1. 1. 1. 1. 0.]\n",
            " [0. 1. 1. 1. 1. 1. 0.]\n",
            " [0. 1. 1. 1. 1. 1. 0.]\n",
            " [0. 1. 1. 1. 1. 1. 0.]\n",
            " [0. 1. 1. 1. 1. 1. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0.]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.ones((5,5))\n",
        "Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GzDBQSQ8qSed"
      },
      "source": [
        "#### 17. What is the result of the following expression? (★☆☆) \n",
        "\n",
        "1.   List item\n",
        "2.   List item\n",
        "\n",
        "\n",
        "(**hint**: NaN = not a number, inf = infinity)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ru2b66sBqSed"
      },
      "source": [
        "```python\n",
        "0 * np.nan\n",
        "np.nan == np.nan\n",
        "np.inf > np.nan\n",
        "np.nan - np.nan\n",
        "0.3 == 3 * 0.1\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 441,
      "metadata": {
        "id": "EYx61WMtqSee",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6c760164-d42a-40af-f057-9da8f7c64a3d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nan\n",
            "False\n",
            "False\n",
            "nan\n",
            "False\n"
          ]
        }
      ],
      "source": [
        "print(0 * np.nan)\n",
        "print(np.nan == np.nan)\n",
        "print(np.inf > np.nan)\n",
        "print(np.nan - np.nan)\n",
        "print(0.3 == 3 * 0.1)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AtN6aQ-4qSee"
      },
      "source": [
        "#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) \n",
        "(**hint**: np.diag)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 442,
      "metadata": {
        "id": "sIJTAJDxqSee",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "64f414f0-4872-45ea-b0c9-05043cd66696"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 0 0 0 0]\n",
            " [1 0 0 0 0]\n",
            " [0 2 0 0 0]\n",
            " [0 0 3 0 0]\n",
            " [0 0 0 4 0]]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "Z = np.diag(1+np.arange(4), k = -1)\n",
        "print (Z)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uhxuMY7_qSef"
      },
      "source": [
        "#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) \n",
        "(**hint**: array\\[::2\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 443,
      "metadata": {
        "id": "lbgKAT9jqSef",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "963b1f48-54fc-4058-ddf1-e9befe8117f7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]\n",
            " [0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]\n",
            " [0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]\n",
            " [0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.zeros ((8,8), dtype=int)\n",
        "Z[1::2, ::2]= 1\n",
        "Z[::2, 1::2] = 1\n",
        "print (Z)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4jrTaTY5qSef"
      },
      "source": [
        "#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? \n",
        "(**hint**: np.unravel_index)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 444,
      "metadata": {
        "id": "O8kdtFcJqSeg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "36b0784a-be27-48df-aa4a-290467b4e93d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(1, 5, 4)\n"
          ]
        }
      ],
      "source": [
        "print (np.unravel_index(100, (6,7,8)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8xqSlWC2qSej"
      },
      "source": [
        "#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) \n",
        "(**hint**: np.tile)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 445,
      "metadata": {
        "id": "Dg8dcYjCqSek",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "aff454f9-bf08-414d-a9e9-a62ee75196bd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]\n",
            " [0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]\n",
            " [0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]\n",
            " [0 1 0 1 0 1 0 1]\n",
            " [1 0 1 0 1 0 1 0]]\n"
          ]
        }
      ],
      "source": [
        "array= np.array([[0,1], [1,0]])\n",
        "Z = np.tile(array,(4,4))\n",
        "print (Z)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U-9ARVj5qSek"
      },
      "source": [
        "#### 22. Normalize a 5x5 random matrix (★☆☆) \n",
        "(**hint**: (x - min) / (max - min))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 446,
      "metadata": {
        "id": "cmfkq1Y3qSel",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f04674ef-6b91-4bf2-a13d-52e86d45e2fc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.21062488 0.         0.67535925 0.84915298 0.33240085]\n",
            " [0.93362464 0.65121129 0.2158119  0.75344267 0.21207059]\n",
            " [0.24012622 0.06595441 0.62077277 0.65642338 0.92112156]\n",
            " [0.43556526 0.12480927 0.4910926  0.26164281 0.86816437]\n",
            " [0.44429771 1.         0.27880351 0.01705336 0.34178296]]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "Z = np.random.random((5,5))\n",
        "Zmax, Zmin = Z.max(), Z.min()\n",
        "Z= (Z-Zmin)/(Zmax-Zmin)\n",
        "print (Z)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "I7IZ0F-5qSel"
      },
      "source": [
        "#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) \n",
        "(**hint**: np.dtype)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 447,
      "metadata": {
        "collapsed": true,
        "id": "TQn2-gAkqSel",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "33981371-bf34-484e-dc13-5bc213bef572"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
            "  \"\"\"\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "Color = np.dtype([(\"r\", np.ubyte, 1), # dtype structured data type\n",
        "                  (\"g\", np.ubyte, 1),\n",
        "                  (\"b\", np.ubyte, 1),\n",
        "                  (\"a\", np.ubyte, 1)])\n",
        "print(Color)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hdmE49xnqSel"
      },
      "source": [
        "#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) \n",
        "(**hint**: np.dot | @)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 448,
      "metadata": {
        "id": "JECZ3fUmqSem",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9551b612-5d8f-4c45-d399-98199690197a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[3. 3.]\n",
            " [3. 3.]\n",
            " [3. 3.]\n",
            " [3. 3.]\n",
            " [3. 3.]]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "Z = np.dot(np.ones((5, 3)), np.ones((3, 2))) # ones((5,3)), the brackets inside are easy to forget\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IMwdhWmmqSem"
      },
      "source": [
        "#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) \n",
        "(**hint**: >, <=)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 449,
      "metadata": {
        "id": "-aSIsSltqSen",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e2e44808-5ddd-4a78-d0f4-1f4118673808"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[ 0  1  2  3 -4 -5 -6 -7  8  9 10]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "Z = np.arange(11)\n",
        "Z[(3 < Z) & (Z < 8)] *= -1\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ijLD8QCCqSen"
      },
      "source": [
        "#### 26. What is the output of the following script? (★☆☆) \n",
        "(**hint**: np.sum)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-sVgTeFyqSeo"
      },
      "source": [
        "```python\n",
        "# Author: Jake VanderPlas\n",
        "\n",
        "print(sum(range(5),-1))\n",
        "from numpy import *\n",
        "print(sum(range(5),-1))\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 449,
      "metadata": {
        "id": "Sg-NREG2qSeo"
      },
      "outputs": [],
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8jlkylSoqSep"
      },
      "source": [
        "#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Rp7FCVg4qSep"
      },
      "source": [
        "```python\n",
        "Z**Z\n",
        "2 << Z >> 2\n",
        "Z <- Z\n",
        "1j*Z\n",
        "Z/1/1\n",
        "Z<Z>Z\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 450,
      "metadata": {
        "id": "Vz3rTdivqSep",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0f6b89bb-d906-4abf-a1da-a42e51bb5e29"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0. 0. 0. 0. 0.]\n",
            "[1. 1. 1. 1. 1.]\n",
            "[False False False False False]\n",
            "[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]\n",
            "[0. 0. 0. 0. 0.]\n"
          ]
        }
      ],
      "source": [
        "Z = zeros(5)\n",
        "print(Z)\n",
        "print(Z**Z)\n",
        "# 2<<Z>>2\n",
        "print(Z<-Z)\n",
        "print(1j*Z)\n",
        "print(Z/1/1)\n",
        "#print(Z<Z>Z)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ML9FstbUqSeq"
      },
      "source": [
        "#### 28. What are the result of the following expressions?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Awduux9AqSeq"
      },
      "source": [
        "```python\n",
        "np.array(0) / np.array(0)\n",
        "np.array(0) // np.array(0)\n",
        "np.array([np.nan]).astype(int).astype(float)\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 451,
      "metadata": {
        "id": "89HQMNNEqSeq",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a7b366a0-657b-44d8-b8ad-5b756033335e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nan\n",
            "0\n",
            "[-9.22337204e+18]\n"
          ]
        }
      ],
      "source": [
        "print(np.array(0) / np.array(0))\n",
        "print(np.array(0) // np.array(0))\n",
        "print(np.array([np.nan]).astype(int).astype(float))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LqlgahFfqSeq"
      },
      "source": [
        "#### 29. How to round away from zero a float array ? (★☆☆) \n",
        "(**hint**: np.uniform, np.copysign, np.ceil, np.abs)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 452,
      "metadata": {
        "id": "lvNAxh5bqSer",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3ab2920f-5390-425b-9045-dfc0c168490c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[ 3.42352345 -4.13002355 -5.18177756 -9.23962421 -8.78215174 -5.9782649\n",
            " -6.96712814 -2.64350696 -2.93694605 -6.19832874]\n",
            "[ 4.  5.  6. 10.  9.  6.  7.  3.  3.  7.]\n",
            "[  4.  -5.  -6. -10.  -9.  -6.  -7.  -3.  -3.  -7.]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.uniform(-10,+10,10)\n",
        "print(Z)\n",
        "print(np.ceil(np.abs(Z)))\n",
        "print (np.copysign(np.ceil(np.abs(Z)), Z))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jbuAeIHGqSer"
      },
      "source": [
        "#### 30. How to find common values between two arrays? (★☆☆) \n",
        "(**hint**: np.intersect1d)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 453,
      "metadata": {
        "id": "NkUJmkdcqSer",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7dcae66b-0697-40ee-eb5b-406e3ed79ac8"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1 3 4]\n"
          ]
        }
      ],
      "source": [
        "Z1 = np.random.randint(0,10,10)\n",
        "Z2 = np.random.randint(0,10,10)\n",
        "print(np.intersect1d(Z1,Z2))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "l1YDNNcNqSes"
      },
      "source": [
        "#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) \n",
        "(**hint**: np.seterr, np.errstate)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 454,
      "metadata": {
        "id": "gPqeY3zYqSes",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fbf67f4d-5f42-4885-a39c-5e00f2b853a7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[inf]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "defaults = np.seterr(all='ignore')\n",
        "Z = np.ones(1)/0\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pVAjRwZGqSes"
      },
      "source": [
        "#### 32. Is the following expressions true? (★☆☆) \n",
        "(**hint**: imaginary number)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S9e40EM2qSes"
      },
      "source": [
        "```python\n",
        "np.sqrt(-1) == np.emath.sqrt(-1)\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 455,
      "metadata": {
        "id": "tLtxCOYkqSet",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "de517340-10f7-41f7-e0a1-9faf76446738"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nan\n",
            "1j\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "False"
            ]
          },
          "metadata": {},
          "execution_count": 455
        }
      ],
      "source": [
        "print(np.sqrt(-1))\n",
        "print(np.emath.sqrt(-1))\n",
        "np.sqrt(-1) == np.emath.sqrt(-1)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oG2di9hbqSet"
      },
      "source": [
        "#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) \n",
        "(**hint**: np.datetime64, np.timedelta64)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 456,
      "metadata": {
        "collapsed": true,
        "id": "fk5OBqOGqSeu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f1055317-ff2a-4cdc-fc46-5379644bd55d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Today:  2022-02-18\n",
            "Yestraday:  2022-02-18\n",
            "Tomorrow:  2022-02-18\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "today = np.datetime64('today', 'D')\n",
        "print(\"Today: \", today)\n",
        "\n",
        "yesterday = np.datetime64('today', 'D')\n",
        "- np.timedelta64(1, 'D')\n",
        "\n",
        "print(\"Yestraday: \", yesterday)\n",
        "\n",
        "tomorrow = np.datetime64('today', 'D')\n",
        "+ np.timedelta64(1, 'D')\n",
        "\n",
        "print(\"Tomorrow: \", tomorrow)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rH63R01gqSeu"
      },
      "source": [
        "#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) \n",
        "(**hint**: np.arange(dtype=datetime64\\['D'\\]))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 457,
      "metadata": {
        "id": "EUpDd_rHqSeu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "064bd6fc-6e5e-47ef-fffe-b4570f401c08"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['2016-07-01' '2016-07-02' '2016-07-03' '2016-07-04' '2016-07-05'\n",
            " '2016-07-06' '2016-07-07' '2016-07-08' '2016-07-09' '2016-07-10'\n",
            " '2016-07-11' '2016-07-12' '2016-07-13' '2016-07-14' '2016-07-15'\n",
            " '2016-07-16' '2016-07-17' '2016-07-18' '2016-07-19' '2016-07-20'\n",
            " '2016-07-21' '2016-07-22' '2016-07-23' '2016-07-24' '2016-07-25'\n",
            " '2016-07-26' '2016-07-27' '2016-07-28' '2016-07-29' '2016-07-30'\n",
            " '2016-07-31']\n"
          ]
        }
      ],
      "source": [
        "G = np.arange('2016-07', '2016-08', dtype='datetime64[D]')\n",
        "print(G)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wFocxu00qSeu"
      },
      "source": [
        "#### 35. How to compute ((A+B)\\*(-A/2)) in place (without copy)? (★★☆) \n",
        "(**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 457,
      "metadata": {
        "id": "JxEJCvgnqSev"
      },
      "outputs": [],
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hriTKVq4qSev"
      },
      "source": [
        "#### 36. Extract the integer part of a random array using 5 different methods (★★☆) \n",
        "(**hint**: %, np.floor, np.ceil, astype, np.trunc)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 458,
      "metadata": {
        "id": "Ors8ah-cqSev",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "50548a96-edd4-4544-f173-22549880b5ff"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-1.5, -1.5, -1.5])"
            ]
          },
          "metadata": {},
          "execution_count": 458
        }
      ],
      "source": [
        "A = np.ones(3)*1\n",
        "B = np.ones(3)*2\n",
        "np.add(A,B,out=B)\n",
        "np.divide(A,2,out=A)\n",
        "np.negative(A,out=A)\n",
        "np.multiply(A,B,out=A)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jJKxkxspqSev"
      },
      "source": [
        "#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) \n",
        "(**hint**: np.arange)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 459,
      "metadata": {
        "id": "u12DMHCXqSew",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "257cd1b6-00d6-4b17-e645-13493591f16e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0. 1. 2. 3. 4.]\n",
            " [0. 1. 2. 3. 4.]\n",
            " [0. 1. 2. 3. 4.]\n",
            " [0. 1. 2. 3. 4.]\n",
            " [0. 1. 2. 3. 4.]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.zeros((5,5))\n",
        "Z += np.arange(5)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EHC76q7sqSew"
      },
      "source": [
        "#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) \n",
        "(**hint**: np.fromiter)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 460,
      "metadata": {
        "id": "o0C7wcl5qSew",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e896be2d-86ff-44df-cb3a-f5d769749749"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]\n"
          ]
        }
      ],
      "source": [
        "def generate():\n",
        "    for x in range(10):\n",
        "        yield x\n",
        "Z = np.fromiter(generate(),dtype=float,count=-1)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YYEZ6mFIqSew"
      },
      "source": [
        "#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) \n",
        "(**hint**: np.linspace)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 461,
      "metadata": {
        "id": "iXDwfTBpqSew",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "96cb962a-e218-4ed5-bf7c-1f2fe1712f71"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.09090909 0.18181818 0.27272727 0.36363636 0.45454545 0.54545455\n",
            " 0.63636364 0.72727273 0.81818182 0.90909091]\n"
          ]
        }
      ],
      "source": [
        "Z = np.linspace(0,1,11,endpoint=False)[1:]\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4sJh_aCjqSex"
      },
      "source": [
        "#### 40. Create a random vector of size 10 and sort it (★★☆) \n",
        "(**hint**: sort)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 462,
      "metadata": {
        "id": "2hmgeRdJqSex",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9efbc386-3dd1-420e-d450-7dfdc3b09796"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.01104576 0.04415953 0.05051554 0.34376544 0.35867466 0.39205137\n",
            " 0.51027507 0.5862507  0.72527026 0.96399312]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.random(10)\n",
        "Z.sort()\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SdN2CXDGqSex"
      },
      "source": [
        "#### 41. How to sum a small array faster than np.sum? (★★☆) \n",
        "(**hint**: np.add.reduce)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 463,
      "metadata": {
        "id": "zWiBNFw9qSex",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "79e3c967-236c-437b-ee4e-ab59f8c11c43"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "45"
            ]
          },
          "metadata": {},
          "execution_count": 463
        }
      ],
      "source": [
        "Z = np.arange(10)\n",
        "np.add.reduce(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LulibPLcqSex"
      },
      "source": [
        "#### 42. Consider two random array A and B, check if they are equal (★★☆) \n",
        "(**hint**: np.allclose, np.array\\_equal)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 464,
      "metadata": {
        "id": "f-_e37oWqSex",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8a8fcbef-6097-428d-cfd9-253236b8efc3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "False\n"
          ]
        }
      ],
      "source": [
        "\n",
        "\n",
        "A = np.random.randint(0,2,5)\n",
        "B = np.random.randint(0,2,5)\n",
        "equal = np.allclose(A,B)\n",
        "print(equal)\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mOxpDWIZqSey"
      },
      "source": [
        "#### 43. Make an array immutable (read-only) (★★☆) \n",
        "(**hint**: flags.writeable)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "SaUBZZvvPorc"
      },
      "execution_count": 464,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 464,
      "metadata": {
        "id": "JqTDlNX2qSey"
      },
      "outputs": [],
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uRZv4x9cqSey"
      },
      "source": [
        "#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) \n",
        "(**hint**: np.sqrt, np.arctan2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 465,
      "metadata": {
        "id": "cr6M07-RqSey",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c5c85163-b038-4d1d-9210-345b22097383"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.83489255 1.02318086 0.75048036 0.04561995 1.28299353 0.91781879\n",
            " 0.54503973 0.54360822 0.04890437 0.55921857]\n",
            "[0.62536146 0.47756645 0.61496782 0.87961057 0.75148775 1.3116579\n",
            " 0.63403775 1.28614662 0.22649756 0.90515273]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.random((10,2))\n",
        "X,Y = Z[:,0], Z[:,1]\n",
        "R = np.sqrt(X**2+Y**2)\n",
        "T = np.arctan2(Y,X)\n",
        "print(R)\n",
        "print(T)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6V-BxJLnqSez"
      },
      "source": [
        "#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) \n",
        "(**hint**: argmax)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 466,
      "metadata": {
        "id": "Up10pW2GqSez",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ada65f55-7d9b-43b2-cdb1-2348571f7ed5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.5450626  0.15414065 0.63297102 0.80289122 0.64540429 0.\n",
            " 0.5384609  0.43190864 0.23522537 0.61175001]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.random(10)\n",
        "Z[Z.argmax()] = 0\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NGHBeotRqSez"
      },
      "source": [
        "#### 46. Create a structured array with `x` and `y` coordinates covering the \\[0,1\\]x\\[0,1\\] area (★★☆) \n",
        "(**hint**: np.meshgrid)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 467,
      "metadata": {
        "id": "l0mxTaP9qSez",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6f765aa9-acad-4b84-dcac-f6e0a1a6fce9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[(0.        , 0.        ) (0.11111111, 0.        )\n",
            "  (0.22222222, 0.        ) (0.33333333, 0.        )\n",
            "  (0.44444444, 0.        ) (0.55555556, 0.        )\n",
            "  (0.66666667, 0.        ) (0.77777778, 0.        )\n",
            "  (0.88888889, 0.        ) (1.        , 0.        )]\n",
            " [(0.        , 0.11111111) (0.11111111, 0.11111111)\n",
            "  (0.22222222, 0.11111111) (0.33333333, 0.11111111)\n",
            "  (0.44444444, 0.11111111) (0.55555556, 0.11111111)\n",
            "  (0.66666667, 0.11111111) (0.77777778, 0.11111111)\n",
            "  (0.88888889, 0.11111111) (1.        , 0.11111111)]\n",
            " [(0.        , 0.22222222) (0.11111111, 0.22222222)\n",
            "  (0.22222222, 0.22222222) (0.33333333, 0.22222222)\n",
            "  (0.44444444, 0.22222222) (0.55555556, 0.22222222)\n",
            "  (0.66666667, 0.22222222) (0.77777778, 0.22222222)\n",
            "  (0.88888889, 0.22222222) (1.        , 0.22222222)]\n",
            " [(0.        , 0.33333333) (0.11111111, 0.33333333)\n",
            "  (0.22222222, 0.33333333) (0.33333333, 0.33333333)\n",
            "  (0.44444444, 0.33333333) (0.55555556, 0.33333333)\n",
            "  (0.66666667, 0.33333333) (0.77777778, 0.33333333)\n",
            "  (0.88888889, 0.33333333) (1.        , 0.33333333)]\n",
            " [(0.        , 0.44444444) (0.11111111, 0.44444444)\n",
            "  (0.22222222, 0.44444444) (0.33333333, 0.44444444)\n",
            "  (0.44444444, 0.44444444) (0.55555556, 0.44444444)\n",
            "  (0.66666667, 0.44444444) (0.77777778, 0.44444444)\n",
            "  (0.88888889, 0.44444444) (1.        , 0.44444444)]\n",
            " [(0.        , 0.55555556) (0.11111111, 0.55555556)\n",
            "  (0.22222222, 0.55555556) (0.33333333, 0.55555556)\n",
            "  (0.44444444, 0.55555556) (0.55555556, 0.55555556)\n",
            "  (0.66666667, 0.55555556) (0.77777778, 0.55555556)\n",
            "  (0.88888889, 0.55555556) (1.        , 0.55555556)]\n",
            " [(0.        , 0.66666667) (0.11111111, 0.66666667)\n",
            "  (0.22222222, 0.66666667) (0.33333333, 0.66666667)\n",
            "  (0.44444444, 0.66666667) (0.55555556, 0.66666667)\n",
            "  (0.66666667, 0.66666667) (0.77777778, 0.66666667)\n",
            "  (0.88888889, 0.66666667) (1.        , 0.66666667)]\n",
            " [(0.        , 0.77777778) (0.11111111, 0.77777778)\n",
            "  (0.22222222, 0.77777778) (0.33333333, 0.77777778)\n",
            "  (0.44444444, 0.77777778) (0.55555556, 0.77777778)\n",
            "  (0.66666667, 0.77777778) (0.77777778, 0.77777778)\n",
            "  (0.88888889, 0.77777778) (1.        , 0.77777778)]\n",
            " [(0.        , 0.88888889) (0.11111111, 0.88888889)\n",
            "  (0.22222222, 0.88888889) (0.33333333, 0.88888889)\n",
            "  (0.44444444, 0.88888889) (0.55555556, 0.88888889)\n",
            "  (0.66666667, 0.88888889) (0.77777778, 0.88888889)\n",
            "  (0.88888889, 0.88888889) (1.        , 0.88888889)]\n",
            " [(0.        , 1.        ) (0.11111111, 1.        )\n",
            "  (0.22222222, 1.        ) (0.33333333, 1.        )\n",
            "  (0.44444444, 1.        ) (0.55555556, 1.        )\n",
            "  (0.66666667, 1.        ) (0.77777778, 1.        )\n",
            "  (0.88888889, 1.        ) (1.        , 1.        )]]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "Z = np.zeros((10,10), [('x',float),('y',float)])\n",
        "Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))\n",
        "print(Z)\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5AlMY7j8qSez"
      },
      "source": [
        "####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) \n",
        "(**hint**: np.subtract.outer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 468,
      "metadata": {
        "id": "8O-5FzhlqSe0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8119a5e5-fa02-41a8-df72-fc9716b4b18f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "3638.163637117973\n"
          ]
        }
      ],
      "source": [
        "X = np.arange(8)\n",
        "Y = X + 0.5\n",
        "C = 1.0 / np.subtract.outer(X, Y)\n",
        "print(np.linalg.det(C))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CNE3a_eSqSe0"
      },
      "source": [
        "#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) \n",
        "(**hint**: np.iinfo, np.finfo, eps)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 469,
      "metadata": {
        "id": "IeAz4hSBqSe0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "57fe3dcb-1bee-444a-d9a7-2ea4a24bbf20"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-128\n",
            "127\n",
            "-2147483648\n",
            "2147483647\n",
            "-9223372036854775808\n",
            "9223372036854775807\n",
            "-3.4028235e+38\n",
            "3.4028235e+38\n",
            "1.1920929e-07\n",
            "-1.7976931348623157e+308\n",
            "1.7976931348623157e+308\n",
            "2.220446049250313e-16\n"
          ]
        }
      ],
      "source": [
        "for dtype in [np.int8, np.int32, np.int64]:\n",
        "   print(np.iinfo(dtype).min)\n",
        "   print(np.iinfo(dtype).max)\n",
        "for dtype in [np.float32, np.float64]:\n",
        "   print(np.finfo(dtype).min)\n",
        "   print(np.finfo(dtype).max)\n",
        "   print(np.finfo(dtype).eps)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QZJYD6D5qSe0"
      },
      "source": [
        "#### 49. How to print all the values of an array? (★★☆) \n",
        "(**hint**: np.set\\_printoptions)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 470,
      "metadata": {
        "id": "J1JpIyvvqSe1",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a3dcc853-af70-45d0-b5ee-901cef8e385f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
            "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]\n"
          ]
        }
      ],
      "source": [
        "np.set_printoptions(threshold=float(\"inf\"))\n",
        "Z = np.zeros((40,40))\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9aOH5Fx9qSe1"
      },
      "source": [
        "#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) \n",
        "(**hint**: argmin)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 471,
      "metadata": {
        "id": "9kAOrAtgqSe2",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "80a83da6-328a-4252-90ea-7a2b6e5cbbf3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "21\n"
          ]
        }
      ],
      "source": [
        "Z = np.arange(100)\n",
        "v = np.random.uniform(0,100)\n",
        "index = (np.abs(Z-v)).argmin()\n",
        "print(Z[index])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hU2s5hDlqSe2"
      },
      "source": [
        "#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) \n",
        "(**hint**: dtype)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 472,
      "metadata": {
        "id": "k9dNjnimqSe2",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "edd19c08-659a-40ac-eb46-04feda374122"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))\n",
            " ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))\n",
            " ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))\n",
            " ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))\n",
            " ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
            "  \"\"\"\n"
          ]
        }
      ],
      "source": [
        "Z = np.zeros(10, [ ('position', [ ('x', float, 1),\n",
        "                                  ('y', float, 1)]),\n",
        "                   ('color',    [ ('r', float, 1),\n",
        "                                  ('g', float, 1),\n",
        "                                  ('b', float, 1)])])\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wn3xvyUfqSe3"
      },
      "source": [
        "#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) \n",
        "(**hint**: np.atleast\\_2d, T, np.sqrt)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 473,
      "metadata": {
        "id": "PnB5wMc-qSe3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e1707dc2-3a02-4c4f-b12e-07c3ba85ece6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.         0.27679601 0.48057264 0.64642059 0.42694392 0.68070682\n",
            "  0.72811432 0.64303075 0.84444271 0.83798651]\n",
            " [0.27679601 0.         0.41739728 0.47287358 0.26882065 0.52167508\n",
            "  0.47367315 0.36730239 0.60305387 0.56958842]\n",
            " [0.48057264 0.41739728 0.         0.2485025  0.6846117  0.25097026\n",
            "  0.78727163 0.60893192 0.92866933 0.63043009]\n",
            " [0.64642059 0.47287358 0.2485025  0.         0.70111353 0.05666733\n",
            "  0.68304479 0.48938281 0.81949757 0.41750825]\n",
            " [0.42694392 0.26882065 0.6846117  0.70111353 0.         0.75518681\n",
            "  0.35505391 0.37525717 0.44084104 0.6476313 ]\n",
            " [0.68070682 0.52167508 0.25097026 0.05666733 0.75518681 0.\n",
            "  0.73864136 0.54482821 0.87448643 0.45801967]\n",
            " [0.72811432 0.47367315 0.78727163 0.68304479 0.35505391 0.73864136\n",
            "  0.         0.19391435 0.14141544 0.41957099]\n",
            " [0.64303075 0.36730239 0.60893192 0.48938281 0.37525717 0.54482821\n",
            "  0.19391435 0.         0.33075699 0.27708031]\n",
            " [0.84444271 0.60305387 0.92866933 0.81949757 0.44084104 0.87448643\n",
            "  0.14141544 0.33075699 0.         0.51886255]\n",
            " [0.83798651 0.56958842 0.63043009 0.41750825 0.6476313  0.45801967\n",
            "  0.41957099 0.27708031 0.51886255 0.        ]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.random((10,2))\n",
        "X,Y = np.atleast_2d(Z[:,0], Z[:,1])\n",
        "D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)\n",
        "print(D)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "D0NmVBm-qSe3"
      },
      "source": [
        "#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? \n",
        "(**hint**: astype(copy=False))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 474,
      "metadata": {
        "id": "1Q6g_1B7qSe3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b721a985-68bc-4c46-fabd-3af9a012376b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[26 58 12 60 67 94 43 64 21 18]\n"
          ]
        }
      ],
      "source": [
        "Z = (np.random.rand(10)*100).astype(np.float32)\n",
        "Y = Z.view(np.int32)\n",
        "Y[:] = Z\n",
        "print(Y)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lNLihz2bqSe3"
      },
      "source": [
        "#### 54. How to read the following file? (★★☆) \n",
        "(**hint**: np.genfromtxt)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "idQzeRttqSe4"
      },
      "source": [
        "```\n",
        "1, 2, 3, 4, 5\n",
        "6,  ,  , 7, 8\n",
        " ,  , 9,10,11\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 475,
      "metadata": {
        "id": "C7hs17MBqSe4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9b16feb6-64e3-4eaf-b1a5-ee3cf0db4733"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 1  2  3  4  5]\n",
            " [ 6 -1 -1  7  8]\n",
            " [-1 -1  9 10 11]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
            "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
            "  \n"
          ]
        }
      ],
      "source": [
        "from io import StringIO\n",
        "s = StringIO('''1, 2, 3, 4, 5\n",
        "                6,  ,  , 7, 8\n",
        "                 ,  , 9,10,11\n",
        "''')\n",
        "Z = np.genfromtxt(s, delimiter=\",\", dtype=np.int)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ia06j0evqSe4"
      },
      "source": [
        "#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) \n",
        "(**hint**: np.ndenumerate, np.ndindex)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 476,
      "metadata": {
        "id": "7zejqDV1qSe4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6b0d0d94-9dec-482f-8a8c-28c5c11e357d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(0, 0) 0\n",
            "(0, 1) 1\n",
            "(0, 2) 2\n",
            "(1, 0) 3\n",
            "(1, 1) 4\n",
            "(1, 2) 5\n",
            "(2, 0) 6\n",
            "(2, 1) 7\n",
            "(2, 2) 8\n",
            "(0, 0) 0\n",
            "(0, 1) 1\n",
            "(0, 2) 2\n",
            "(1, 0) 3\n",
            "(1, 1) 4\n",
            "(1, 2) 5\n",
            "(2, 0) 6\n",
            "(2, 1) 7\n",
            "(2, 2) 8\n"
          ]
        }
      ],
      "source": [
        "Z = np.arange(9).reshape(3,3)\n",
        "for index, value in np.ndenumerate(Z):\n",
        "    print(index, value)\n",
        "for index in np.ndindex(Z.shape):\n",
        "    print(index, Z[index])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nA4KNxv6qSe4"
      },
      "source": [
        "#### 56. Generate a generic 2D Gaussian-like array (★★☆) \n",
        "(**hint**: np.meshgrid, np.exp)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 477,
      "metadata": {
        "id": "dGoqqbT1qSe5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "35006f31-1614-4548-b24a-9015f1cfb844"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.36787944 0.44822088 0.51979489 0.57375342 0.60279818 0.60279818\n",
            "  0.57375342 0.51979489 0.44822088 0.36787944]\n",
            " [0.44822088 0.54610814 0.63331324 0.69905581 0.73444367 0.73444367\n",
            "  0.69905581 0.63331324 0.54610814 0.44822088]\n",
            " [0.51979489 0.63331324 0.73444367 0.81068432 0.85172308 0.85172308\n",
            "  0.81068432 0.73444367 0.63331324 0.51979489]\n",
            " [0.57375342 0.69905581 0.81068432 0.89483932 0.9401382  0.9401382\n",
            "  0.89483932 0.81068432 0.69905581 0.57375342]\n",
            " [0.60279818 0.73444367 0.85172308 0.9401382  0.98773022 0.98773022\n",
            "  0.9401382  0.85172308 0.73444367 0.60279818]\n",
            " [0.60279818 0.73444367 0.85172308 0.9401382  0.98773022 0.98773022\n",
            "  0.9401382  0.85172308 0.73444367 0.60279818]\n",
            " [0.57375342 0.69905581 0.81068432 0.89483932 0.9401382  0.9401382\n",
            "  0.89483932 0.81068432 0.69905581 0.57375342]\n",
            " [0.51979489 0.63331324 0.73444367 0.81068432 0.85172308 0.85172308\n",
            "  0.81068432 0.73444367 0.63331324 0.51979489]\n",
            " [0.44822088 0.54610814 0.63331324 0.69905581 0.73444367 0.73444367\n",
            "  0.69905581 0.63331324 0.54610814 0.44822088]\n",
            " [0.36787944 0.44822088 0.51979489 0.57375342 0.60279818 0.60279818\n",
            "  0.57375342 0.51979489 0.44822088 0.36787944]]\n"
          ]
        }
      ],
      "source": [
        "X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))\n",
        "D = np.sqrt(X*X+Y*Y)\n",
        "sigma, mu = 1.0, 0.0\n",
        "G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )\n",
        "print(G)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tQlQraHYqSe5"
      },
      "source": [
        "#### 57. How to randomly place p elements in a 2D array? (★★☆) \n",
        "(**hint**: np.put, np.random.choice)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 478,
      "metadata": {
        "id": "c08DwgudqSe5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7106ffa1-d08e-4f6f-f4e7-6f8da53a4c14"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]\n",
            " [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]\n"
          ]
        }
      ],
      "source": [
        "n = 10\n",
        "p = 3\n",
        "Z = np.zeros((n,n))\n",
        "np.put(Z, np.random.choice(range(n*n), p, replace=False),1)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Tnx9DEJWqSe5"
      },
      "source": [
        "#### 58. Subtract the mean of each row of a matrix (★★☆) \n",
        "(**hint**: mean(axis=,keepdims=))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 479,
      "metadata": {
        "id": "5lLlLMlMqSe5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5c4baa4c-083b-408d-ccfe-91bd5d631faf"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-3.82375918e-01  1.52683473e-01  5.22217809e-01 -2.52779632e-01\n",
            "  -8.81622261e-03  2.59231660e-02 -8.23728222e-02  4.98425289e-01\n",
            "  -3.89688924e-01 -8.32162182e-02]\n",
            " [ 3.71562100e-01 -2.60931896e-01 -3.59062459e-01  1.47085069e-01\n",
            "   4.39384677e-01  4.20037249e-02  1.70786988e-01 -2.29039194e-01\n",
            "  -3.42996505e-01  2.12074958e-02]\n",
            " [-3.07938639e-01  2.41167493e-01  3.73647045e-01 -1.54644874e-01\n",
            "   3.26511328e-01 -1.10766588e-01 -1.60840063e-01 -1.55889574e-01\n",
            "  -6.39730158e-02  1.27268892e-02]\n",
            " [ 2.47139226e-01  1.80430852e-01  4.62185866e-01 -4.78748477e-01\n",
            "   3.93766182e-01 -3.20864973e-02 -3.97144645e-01  1.07706433e-04\n",
            "  -3.29494808e-01 -4.61554055e-02]\n",
            " [ 7.51791561e-02  2.22917670e-01  1.92410970e-01 -4.00774062e-01\n",
            "   1.08447668e-01  3.28178624e-01  3.40267550e-01 -3.89560460e-01\n",
            "  -4.49487552e-01 -2.75795661e-02]]\n"
          ]
        }
      ],
      "source": [
        "X = np.random.rand(5, 10)\n",
        "\n",
        "# Recent versions of numpy\n",
        "Y = X - X.mean(axis=1, keepdims=True)\n",
        "\n",
        "# Older versions of numpy\n",
        "Y = X - X.mean(axis=1).reshape(-1, 1)\n",
        "\n",
        "print(Y)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eSlEgra5qSe6"
      },
      "source": [
        "#### 59. How to sort an array by the nth column? (★★☆) \n",
        "(**hint**: argsort)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 480,
      "metadata": {
        "id": "6P6IKm4WqSe6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "add8313d-727d-41a8-9a25-a433f1ddf9de"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[4 4 0]\n",
            " [9 7 9]\n",
            " [7 1 7]]\n",
            "[[7 1 7]\n",
            " [4 4 0]\n",
            " [9 7 9]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.randint(0,10,(3,3))\n",
        "print(Z)\n",
        "print(Z[Z[:,1].argsort()])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uw801dfBqSe6"
      },
      "source": [
        "#### 60. How to tell if a given 2D array has null columns? (★★☆) \n",
        "(**hint**: any, ~)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 481,
      "metadata": {
        "id": "ht25Xlk0qSe6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0f83f6c3-49b4-439e-a5b1-6797410874a1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "False\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.randint(0,3,(3,10))\n",
        "print((~Z.any(axis=0)).any())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6WYnRiwLqSe6"
      },
      "source": [
        "#### 61. Find the nearest value from a given value in an array (★★☆) \n",
        "(**hint**: np.abs, argmin, flat)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 482,
      "metadata": {
        "id": "ojLZqqCwqSe7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d945927e-25d3-4c46-a33a-b2199d7ae0af"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.4867438613777887\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.uniform(0,1,10)\n",
        "z = 0.5\n",
        "m = Z.flat[np.abs(Z - z).argmin()]\n",
        "print(m)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2esIrWYjqSe7"
      },
      "source": [
        "#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) \n",
        "(**hint**: np.nditer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 483,
      "metadata": {
        "id": "yOJctCmFqSe7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ded46e06-f7c8-4607-e662-098445ec11dc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 1 2]\n",
            " [1 2 3]\n",
            " [2 3 4]]\n"
          ]
        }
      ],
      "source": [
        "A = np.arange(3).reshape(3,1)\n",
        "B = np.arange(3).reshape(1,3)\n",
        "it = np.nditer([A,B,None])\n",
        "for x,y,z in it: z[...] = x + y\n",
        "print(it.operands[2])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2IJHIFXqqSe7"
      },
      "source": [
        "#### 63. Create an array class that has a name attribute (★★☆) \n",
        "(**hint**: class method)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 484,
      "metadata": {
        "id": "yCXCYQNbqSe7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b5c5de6f-6f6f-4c8d-d093-06000ebc4840"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "range_10\n"
          ]
        }
      ],
      "source": [
        "class NamedArray(np.ndarray):\n",
        "    def __new__(cls, array, name=\"no name\"):\n",
        "        obj = np.asarray(array).view(cls)\n",
        "        obj.name = name\n",
        "        return obj\n",
        "    def __array_finalize__(self, obj):\n",
        "        if obj is None: return\n",
        "        self.info = getattr(obj, 'name', \"no name\")\n",
        "\n",
        "Z = NamedArray(np.arange(10), \"range_10\")\n",
        "print (Z.name)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "etPSyLnGqSe8"
      },
      "source": [
        "#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) \n",
        "(**hint**: np.bincount | np.add.at)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 485,
      "metadata": {
        "id": "RpmtsKhEqSe8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2bc39464-7d22-4bab-b71c-cf6bc350fe65"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2. 5. 2. 4. 3. 4. 3. 2. 3. 2.]\n",
            "[3. 9. 3. 7. 5. 7. 5. 3. 5. 3.]\n"
          ]
        }
      ],
      "source": [
        "Z = np.ones(10)\n",
        "I = np.random.randint(0,len(Z),20)\n",
        "Z += np.bincount(I, minlength=len(Z))\n",
        "print(Z)\n",
        "\n",
        "# Another solution\n",
        "# Author: Bartosz Telenczuk\n",
        "np.add.at(Z, I, 1)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Z5-Dj7PAqSe8"
      },
      "source": [
        "#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) \n",
        "(**hint**: np.bincount)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 486,
      "metadata": {
        "id": "JWTpoXjiqSe8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d538e489-9d21-42b5-bd1c-84e9d39afebd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0. 7. 0. 6. 5. 0. 0. 0. 0. 3.]\n"
          ]
        }
      ],
      "source": [
        "X = [1,2,3,4,5,6]\n",
        "I = [1,3,9,3,4,1]\n",
        "F = np.bincount(I,X)\n",
        "print(F)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zLkMfGO4qSe8"
      },
      "source": [
        "#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) \n",
        "(**hint**: np.unique)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 487,
      "metadata": {
        "id": "6XvGa8fTqSe9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "cd9b6982-997b-4ff6-bfe2-aeac0f2d0db6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "64\n"
          ]
        }
      ],
      "source": [
        "w, h = 256, 256\n",
        "I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)\n",
        "colors = np.unique(I.reshape(-1, 3), axis=0)\n",
        "n = len(colors)\n",
        "print(n)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qZNA2VxmqSe9"
      },
      "source": [
        "#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) \n",
        "(**hint**: sum(axis=(-2,-1)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 488,
      "metadata": {
        "id": "B0P3_dhCqSe9",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bd0bbf30-5f61-4980-9d15-db127c6608ec"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[44 40 72 57]\n",
            " [53 48 50 52]\n",
            " [53 58 64 56]]\n",
            "[[44 40 72 57]\n",
            " [53 48 50 52]\n",
            " [53 58 64 56]]\n"
          ]
        }
      ],
      "source": [
        "A = np.random.randint(0,10,(3,4,3,4))\n",
        "sum = A.sum(axis=(-2,-1))\n",
        "print(sum)\n",
        "sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)\n",
        "print(sum)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VDoINBgMqSe9"
      },
      "source": [
        "#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) \n",
        "(**hint**: np.bincount)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 489,
      "metadata": {
        "id": "8QSK72cUqSe-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1f20f0a1-ac14-4f53-899f-d0a321a84f4d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.59061428 0.55937403 0.4567408  0.44574723 0.57907371 0.58479702\n",
            " 0.43993687 0.48659026 0.73196797 0.68728834]\n"
          ]
        }
      ],
      "source": [
        "D = np.random.uniform(0,1,100)\n",
        "S = np.random.randint(0,10,100)\n",
        "D_sums = np.bincount(S, weights=D)\n",
        "D_counts = np.bincount(S)\n",
        "D_means = D_sums / D_counts\n",
        "print(D_means)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BYauDAUMqSe-"
      },
      "source": [
        "#### 69. How to get the diagonal of a dot product? (★★★) \n",
        "(**hint**: np.diag)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 490,
      "metadata": {
        "id": "pHO51wUnqSe-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0bc4af4c-02a3-4372-9a31-7b0304c0ae5a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0.67825147, 1.06936271, 1.20033984, 0.60156927, 1.20678833])"
            ]
          },
          "metadata": {},
          "execution_count": 490
        }
      ],
      "source": [
        "A = np.random.uniform(0,1,(5,5))\n",
        "B = np.random.uniform(0,1,(5,5))\n",
        "\n",
        "np.diag(np.dot(A, B))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Rljbu58GqSe-"
      },
      "source": [
        "#### 70. Consider the vector \\[1, 2, 3, 4, 5\\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) \n",
        "(**hint**: array\\[::4\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 491,
      "metadata": {
        "id": "LU15nEynqSe_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bfb8d408-d0a3-4adc-b348-5b0e0109b959"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1. 0. 0. 0. 2. 0. 0. 0. 3. 0. 0. 0. 4. 0. 0. 0. 5.]\n"
          ]
        }
      ],
      "source": [
        "Z = np.array([1,2,3,4,5])\n",
        "nz = 3\n",
        "Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))\n",
        "Z0[::nz+1] = Z\n",
        "print(Z0)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LCG9jZmKqSe_"
      },
      "source": [
        "#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) \n",
        "(**hint**: array\\[:, :, None\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 492,
      "metadata": {
        "collapsed": true,
        "id": "wff03StgqSe_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "422a6d3a-bafe-425a-da5f-81e0683645d4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[[2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]]\n",
            "\n",
            " [[2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]]\n",
            "\n",
            " [[2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]]\n",
            "\n",
            " [[2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]]\n",
            "\n",
            " [[2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]\n",
            "  [2. 2. 2.]]]\n"
          ]
        }
      ],
      "source": [
        "A = np.ones((5,5,3))\n",
        "B = 2*np.ones((5,5))\n",
        "print(A * B[:,:,None])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GF864YQMqSe_"
      },
      "source": [
        "#### 72. How to swap two rows of an array? (★★★) \n",
        "(**hint**: array\\[\\[\\]\\] = array\\[\\[\\]\\])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 493,
      "metadata": {
        "id": "SpSfqwaMqSfA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "19425aa1-6ad9-47d9-f4c2-ce4b2b915780"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 5  6  7  8  9]\n",
            " [ 0  1  2  3  4]\n",
            " [10 11 12 13 14]\n",
            " [15 16 17 18 19]\n",
            " [20 21 22 23 24]]\n"
          ]
        }
      ],
      "source": [
        "A = np.arange(25).reshape(5,5)\n",
        "A[[0,1]] = A[[1,0]]\n",
        "print(A)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DasHN_PHqSfA"
      },
      "source": [
        "#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) \n",
        "(**hint**: repeat, np.roll, np.sort, view, np.unique)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 494,
      "metadata": {
        "id": "9VPQEn5oqSfA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "150477e4-0246-4671-8fbf-23c0adb6a98f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[( 2, 51) ( 2, 96) ( 6, 91) ( 6, 99) (12, 30) (12, 39) (12, 63) (12, 84)\n",
            " (15, 72) (15, 77) (17, 57) (17, 95) (19, 79) (19, 82) (30, 35) (30, 38)\n",
            " (30, 63) (31, 62) (31, 70) (35, 38) (39, 84) (47, 51) (47, 83) (51, 83)\n",
            " (51, 96) (57, 95) (62, 70) (72, 77) (79, 82) (91, 99)]\n"
          ]
        }
      ],
      "source": [
        "faces = np.random.randint(0,100,(10,3))\n",
        "F = np.roll(faces.repeat(2,axis=1),-1,axis=1)\n",
        "F = F.reshape(len(F)*3,2)\n",
        "F = np.sort(F,axis=1)\n",
        "G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )\n",
        "G = np.unique(G)\n",
        "print(G)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tJr0v2SXqSfA"
      },
      "source": [
        "#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) \n",
        "(**hint**: np.repeat)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 495,
      "metadata": {
        "id": "6ct0vYwTqSfA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f42f2c7f-62ba-4951-bfb4-8eb095a9e477"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1 1 2 3 4 4 6]\n"
          ]
        }
      ],
      "source": [
        "C = np.bincount([1,1,2,3,4,4,6])\n",
        "A = np.repeat(np.arange(len(C)), C)\n",
        "print(A)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S_F9cH2NqSfB"
      },
      "source": [
        "#### 75. How to compute averages using a sliding window over an array? (★★★) \n",
        "(**hint**: np.cumsum)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 496,
      "metadata": {
        "id": "miNz9nhiqSfB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3f705e8f-fbe2-47e5-9727-53afda6d7c26"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "def moving_average(a, n=3) :\n",
        "    ret = np.cumsum(a, dtype=float)\n",
        "    ret[n:] = ret[n:] - ret[:-n]\n",
        "    return ret[n - 1:] / n\n",
        "Z = np.arange(20)\n",
        "print(moving_average(Z, n=3))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SEJlrDCPqSfB"
      },
      "source": [
        "#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\\[0\\],Z\\[1\\],Z\\[2\\]) and each subsequent row is  shifted by 1 (last row should be (Z\\[-3\\],Z\\[-2\\],Z\\[-1\\]) (★★★) \n",
        "(**hint**: from numpy.lib import stride_tricks)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 497,
      "metadata": {
        "id": "8e7eKDhPqSfB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f56b5459-ad0d-4053-b17a-cf282b3e9bd2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 1 2]\n",
            " [1 2 3]\n",
            " [2 3 4]\n",
            " [3 4 5]\n",
            " [4 5 6]\n",
            " [5 6 7]\n",
            " [6 7 8]\n",
            " [7 8 9]]\n"
          ]
        }
      ],
      "source": [
        "from numpy.lib import stride_tricks\n",
        "\n",
        "def rolling(a, window):\n",
        "    shape = (a.size - window + 1, window)\n",
        "    strides = (a.strides[0], a.strides[0])\n",
        "    return stride_tricks.as_strided(a, shape=shape, strides=strides)\n",
        "Z = rolling(np.arange(10), 3)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "92rl4KkKqSfC"
      },
      "source": [
        "#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) \n",
        "(**hint**: np.logical_not, np.negative)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 498,
      "metadata": {
        "id": "dIyHOlBZqSfC",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6e4f5539-1ae1-4790-e980-d9f0f56bc4bc"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-0.17059394, -0.16130811, -0.6911975 ,  0.95784757,  0.08972093,\n",
              "       -0.6927631 ,  0.0616502 , -0.61543163,  0.18662433,  0.67978246,\n",
              "        0.21823849, -0.94951364, -0.38073731,  0.71103661,  0.71672138,\n",
              "        0.46906631, -0.01494213,  0.69743575, -0.41886326, -0.9358425 ,\n",
              "       -0.68244169,  0.48071732, -0.08637256,  0.5845908 ,  0.44977473,\n",
              "        0.15858153, -0.33033969,  0.55916444,  0.89807409,  0.00677635,\n",
              "       -0.73312166, -0.49177701, -0.68103123, -0.87768671,  0.93801427,\n",
              "       -0.38776413, -0.37595906,  0.20673631,  0.24004366, -0.41792897,\n",
              "        0.67987066, -0.93095137,  0.72720759,  0.87848734, -0.84519103,\n",
              "        0.46284137,  0.00483072,  0.73721082, -0.57770082,  0.27809576,\n",
              "        0.16427643, -0.33670948,  0.41429682, -0.68390813, -0.81329201,\n",
              "       -0.46149884, -0.33901896,  0.04264122, -0.47692197,  0.73210581,\n",
              "       -0.55276392, -0.66908737, -0.8382187 , -0.12704125,  0.09574607,\n",
              "       -0.09869334,  0.95056504, -0.64183518, -0.12341194,  0.19549588,\n",
              "        0.25553611, -0.99751655, -0.08936155, -0.28003145, -0.50957456,\n",
              "        0.71109243, -0.56861385, -0.59599021,  0.2007981 , -0.56761772,\n",
              "        0.34513833, -0.83764195, -0.47069737, -0.47007324,  0.56524147,\n",
              "        0.20061952,  0.01633293, -0.70545054, -0.7696553 ,  0.59493249,\n",
              "        0.45698517,  0.72773917, -0.54948814, -0.32503689,  0.97614871,\n",
              "        0.44877714,  0.47458145, -0.18757612,  0.67770966,  0.71232472])"
            ]
          },
          "metadata": {},
          "execution_count": 498
        }
      ],
      "source": [
        "Z = np.random.randint(0,2,100)\n",
        "np.logical_not(Z, out=Z)\n",
        "\n",
        "Z = np.random.uniform(-1.0,1.0,100)\n",
        "np.negative(Z, out=Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1k62zI1vqSfC"
      },
      "source": [
        "#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\\[i\\],P1\\[i\\])? (★★★)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 499,
      "metadata": {
        "id": "AqgZLqVoqSfD",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1bde5f53-701a-483a-ab45-08930b9bcf1c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2.48316354 5.02085313 6.52489971 8.36440007 7.39514604 5.78276551\n",
            " 3.36518744 5.89110512 4.96983864 1.21950713]\n"
          ]
        }
      ],
      "source": [
        "def distance(P0, P1, p):\n",
        "    T = P1 - P0\n",
        "    L = (T**2).sum(axis=1)\n",
        "    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L\n",
        "    U = U.reshape(len(U),1)\n",
        "    D = P0 + U*T - p\n",
        "    return np.sqrt((D**2).sum(axis=1))\n",
        "\n",
        "P0 = np.random.uniform(-10,10,(10,2))\n",
        "P1 = np.random.uniform(-10,10,(10,2))\n",
        "p  = np.random.uniform(-10,10,( 1,2))\n",
        "print(distance(P0, P1, p))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pnhQb9ArqSfD"
      },
      "source": [
        "#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\\[j\\]) to each line i (P0\\[i\\],P1\\[i\\])? (★★★)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 500,
      "metadata": {
        "id": "9cHYdEkzqSfE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "07bcc960-26e4-4ac2-8bc1-083fb9afdfb4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[14.49511045  5.5540516  14.3938817   4.82569679  9.08136866  0.71316361\n",
            "  14.13532081  6.86610364 11.44738216 12.7346276 ]\n",
            " [ 4.9132028   5.07105874  3.39666355 10.88337498 10.47436512  0.24254256\n",
            "   5.50949267  4.77229184  3.33472012  5.82203207]\n",
            " [ 0.21361347  5.59322253  1.80513664  9.53831134  5.90688113  3.33664767\n",
            "   1.60109797  4.96286125  8.06741412  0.76579435]\n",
            " [ 7.30761855  6.95504887  6.26431701  3.24715326  2.12857816  6.09253189\n",
            "   8.32301734  7.74317996  3.6704415   4.89212447]\n",
            " [ 5.70512633 11.79320805  3.1445618   6.49071144  0.82379313 12.15484472\n",
            "   8.26784673 12.33842644  2.1115445   2.32050155]\n",
            " [ 1.64895262  4.60754461  2.68389215  1.97185478  6.75745564  8.3976709\n",
            "   0.04204422  4.85685232  2.35940462  4.1362367 ]\n",
            " [ 5.76862343 13.5182809   2.71454007  7.88496719  0.97941761 13.97420388\n",
            "   8.79550437 14.01971755  3.59689123  2.07380096]\n",
            " [ 1.5602216   2.76200855  0.45991997  7.79333997  7.07024579  0.43129134\n",
            "   1.9858592   2.30137826  4.31278462  2.7022884 ]\n",
            " [ 3.43716465  2.48027995  2.55857671  8.27239639  8.89046012  1.74566534\n",
            "   3.5267484   2.15412244  2.35716066  4.73498111]\n",
            " [15.6803726   6.58942777 15.36871042  6.09210246 10.29658911  1.18682388\n",
            "  15.44098293  7.9525919  11.59711606 13.79758746]]\n"
          ]
        }
      ],
      "source": [
        "P0 = np.random.uniform(-10, 10, (10,2))\n",
        "P1 = np.random.uniform(-10,10,(10,2))\n",
        "p = np.random.uniform(-10, 10, (10,2))\n",
        "print(np.array([distance(P0,P1,p_i) for p_i in p]))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qD7z0SElqSfE"
      },
      "source": [
        "#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) \n",
        "(**hint**: minimum, maximum)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 521,
      "metadata": {
        "id": "azYl82wVqSfF",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4ff32ca6-8a1f-4c21-d0f4-7edb4280aea2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[6 1 9 2 5 6 2 6 6 4]\n",
            " [3 5 8 4 5 7 8 2 9 5]\n",
            " [3 8 2 2 3 0 3 4 2 8]\n",
            " [9 3 1 0 8 0 8 1 8 6]\n",
            " [6 2 4 0 5 5 8 2 4 9]\n",
            " [9 3 3 6 1 3 4 7 2 6]\n",
            " [7 4 6 9 2 1 5 2 1 0]\n",
            " [2 1 6 6 6 9 3 0 9 1]\n",
            " [8 3 9 0 1 3 6 8 6 7]\n",
            " [8 6 6 1 5 3 0 6 6 6]]\n",
            "[[0 0 0 0 0]\n",
            " [0 6 1 9 2]\n",
            " [0 3 5 8 4]\n",
            " [0 3 8 2 2]\n",
            " [0 9 3 1 0]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:23: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.randint(0,10,(10,10))\n",
        "shape = (5,5)\n",
        "fill  = 0\n",
        "position = (1,1)\n",
        "\n",
        "R = np.ones(shape, dtype=Z.dtype)*fill\n",
        "P  = np.array(list(position)).astype(int)\n",
        "Rs = np.array(list(R.shape)).astype(int)\n",
        "Zs = np.array(list(Z.shape)).astype(int)\n",
        "\n",
        "R_start = np.zeros((len(shape),)).astype(int)\n",
        "R_stop  = np.array(list(shape)).astype(int)\n",
        "Z_start = (P-Rs//2)\n",
        "Z_stop  = (P+Rs//2)+Rs%2\n",
        "\n",
        "R_start = (R_start - np.minimum(Z_start,0)).tolist()\n",
        "Z_start = (np.maximum(Z_start,0)).tolist()\n",
        "R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()\n",
        "Z_stop = (np.minimum(Z_stop,Zs)).tolist()\n",
        "\n",
        "r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]\n",
        "z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]\n",
        "R[r] = Z[z]\n",
        "print(Z)\n",
        "print(R)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S_toTVWNqSfF"
      },
      "source": [
        "#### 81. Consider an array Z = \\[1,2,3,4,5,6,7,8,9,10,11,12,13,14\\], how to generate an array R = \\[\\[1,2,3,4\\], \\[2,3,4,5\\], \\[3,4,5,6\\], ..., \\[11,12,13,14\\]\\]? (★★★) \n",
        "(**hint**: stride\\_tricks.as\\_strided)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 501,
      "metadata": {
        "id": "vpqeO5wBqSfG",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e2aa87a4-a25c-40c2-bc31-dc6f34e4a10e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 1  2  3  4]\n",
            " [ 2  3  4  5]\n",
            " [ 3  4  5  6]\n",
            " [ 4  5  6  7]\n",
            " [ 5  6  7  8]\n",
            " [ 6  7  8  9]\n",
            " [ 7  8  9 10]\n",
            " [ 8  9 10 11]\n",
            " [ 9 10 11 12]\n",
            " [10 11 12 13]\n",
            " [11 12 13 14]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.arange(1,15,dtype=np.uint32)\n",
        "R = stride_tricks.as_strided(Z,(11,4),(4,4))\n",
        "print(R)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "guSK31cGqSfG"
      },
      "source": [
        "#### 82. Compute a matrix rank (★★★) \n",
        "(**hint**: np.linalg.svd) (suggestion: np.linalg.svd)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 502,
      "metadata": {
        "id": "Lepb-WiXqSfG",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "89598893-c8fd-44ba-c3d1-17bc9cd2813d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "10\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.uniform(0,1,(10,10))\n",
        "U, S, V = np.linalg.svd(Z) \n",
        "rank = np.sum(S > 1e-10)\n",
        "print(rank)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LHHUkjDUqSfH"
      },
      "source": [
        "#### 83. How to find the most frequent value in an array? \n",
        "(**hint**: np.bincount, argmax)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 503,
      "metadata": {
        "id": "sC9Y5H30qSfH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8c6514ec-b242-465d-feb6-737385e2a793"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "4\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "Z = np.random.randint(0,10,50)\n",
        "print(np.bincount(Z).argmax())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CsgthtW1qSfH"
      },
      "source": [
        "#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) \n",
        "(**hint**: stride\\_tricks.as\\_strided)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 504,
      "metadata": {
        "id": "D3yu1LQeqSfI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e2ecf6cf-903c-4cf2-ff2a-f34cc75553ad"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[[[2 2 2]\n",
            "   [2 4 4]\n",
            "   [1 0 2]]\n",
            "\n",
            "  [[2 2 1]\n",
            "   [4 4 4]\n",
            "   [0 2 3]]\n",
            "\n",
            "  [[2 1 4]\n",
            "   [4 4 0]\n",
            "   [2 3 0]]\n",
            "\n",
            "  [[1 4 0]\n",
            "   [4 0 4]\n",
            "   [3 0 0]]\n",
            "\n",
            "  [[4 0 1]\n",
            "   [0 4 2]\n",
            "   [0 0 4]]\n",
            "\n",
            "  [[0 1 1]\n",
            "   [4 2 4]\n",
            "   [0 4 4]]\n",
            "\n",
            "  [[1 1 0]\n",
            "   [2 4 1]\n",
            "   [4 4 2]]\n",
            "\n",
            "  [[1 0 0]\n",
            "   [4 1 1]\n",
            "   [4 2 0]]]\n",
            "\n",
            "\n",
            " [[[2 4 4]\n",
            "   [1 0 2]\n",
            "   [0 3 1]]\n",
            "\n",
            "  [[4 4 4]\n",
            "   [0 2 3]\n",
            "   [3 1 1]]\n",
            "\n",
            "  [[4 4 0]\n",
            "   [2 3 0]\n",
            "   [1 1 3]]\n",
            "\n",
            "  [[4 0 4]\n",
            "   [3 0 0]\n",
            "   [1 3 2]]\n",
            "\n",
            "  [[0 4 2]\n",
            "   [0 0 4]\n",
            "   [3 2 3]]\n",
            "\n",
            "  [[4 2 4]\n",
            "   [0 4 4]\n",
            "   [2 3 3]]\n",
            "\n",
            "  [[2 4 1]\n",
            "   [4 4 2]\n",
            "   [3 3 0]]\n",
            "\n",
            "  [[4 1 1]\n",
            "   [4 2 0]\n",
            "   [3 0 1]]]\n",
            "\n",
            "\n",
            " [[[1 0 2]\n",
            "   [0 3 1]\n",
            "   [2 0 3]]\n",
            "\n",
            "  [[0 2 3]\n",
            "   [3 1 1]\n",
            "   [0 3 0]]\n",
            "\n",
            "  [[2 3 0]\n",
            "   [1 1 3]\n",
            "   [3 0 4]]\n",
            "\n",
            "  [[3 0 0]\n",
            "   [1 3 2]\n",
            "   [0 4 4]]\n",
            "\n",
            "  [[0 0 4]\n",
            "   [3 2 3]\n",
            "   [4 4 1]]\n",
            "\n",
            "  [[0 4 4]\n",
            "   [2 3 3]\n",
            "   [4 1 2]]\n",
            "\n",
            "  [[4 4 2]\n",
            "   [3 3 0]\n",
            "   [1 2 1]]\n",
            "\n",
            "  [[4 2 0]\n",
            "   [3 0 1]\n",
            "   [2 1 2]]]\n",
            "\n",
            "\n",
            " [[[0 3 1]\n",
            "   [2 0 3]\n",
            "   [1 1 3]]\n",
            "\n",
            "  [[3 1 1]\n",
            "   [0 3 0]\n",
            "   [1 3 3]]\n",
            "\n",
            "  [[1 1 3]\n",
            "   [3 0 4]\n",
            "   [3 3 3]]\n",
            "\n",
            "  [[1 3 2]\n",
            "   [0 4 4]\n",
            "   [3 3 1]]\n",
            "\n",
            "  [[3 2 3]\n",
            "   [4 4 1]\n",
            "   [3 1 4]]\n",
            "\n",
            "  [[2 3 3]\n",
            "   [4 1 2]\n",
            "   [1 4 3]]\n",
            "\n",
            "  [[3 3 0]\n",
            "   [1 2 1]\n",
            "   [4 3 4]]\n",
            "\n",
            "  [[3 0 1]\n",
            "   [2 1 2]\n",
            "   [3 4 0]]]\n",
            "\n",
            "\n",
            " [[[2 0 3]\n",
            "   [1 1 3]\n",
            "   [1 3 2]]\n",
            "\n",
            "  [[0 3 0]\n",
            "   [1 3 3]\n",
            "   [3 2 2]]\n",
            "\n",
            "  [[3 0 4]\n",
            "   [3 3 3]\n",
            "   [2 2 4]]\n",
            "\n",
            "  [[0 4 4]\n",
            "   [3 3 1]\n",
            "   [2 4 3]]\n",
            "\n",
            "  [[4 4 1]\n",
            "   [3 1 4]\n",
            "   [4 3 4]]\n",
            "\n",
            "  [[4 1 2]\n",
            "   [1 4 3]\n",
            "   [3 4 4]]\n",
            "\n",
            "  [[1 2 1]\n",
            "   [4 3 4]\n",
            "   [4 4 2]]\n",
            "\n",
            "  [[2 1 2]\n",
            "   [3 4 0]\n",
            "   [4 2 0]]]\n",
            "\n",
            "\n",
            " [[[1 1 3]\n",
            "   [1 3 2]\n",
            "   [1 3 2]]\n",
            "\n",
            "  [[1 3 3]\n",
            "   [3 2 2]\n",
            "   [3 2 3]]\n",
            "\n",
            "  [[3 3 3]\n",
            "   [2 2 4]\n",
            "   [2 3 3]]\n",
            "\n",
            "  [[3 3 1]\n",
            "   [2 4 3]\n",
            "   [3 3 0]]\n",
            "\n",
            "  [[3 1 4]\n",
            "   [4 3 4]\n",
            "   [3 0 2]]\n",
            "\n",
            "  [[1 4 3]\n",
            "   [3 4 4]\n",
            "   [0 2 3]]\n",
            "\n",
            "  [[4 3 4]\n",
            "   [4 4 2]\n",
            "   [2 3 4]]\n",
            "\n",
            "  [[3 4 0]\n",
            "   [4 2 0]\n",
            "   [3 4 1]]]\n",
            "\n",
            "\n",
            " [[[1 3 2]\n",
            "   [1 3 2]\n",
            "   [3 4 1]]\n",
            "\n",
            "  [[3 2 2]\n",
            "   [3 2 3]\n",
            "   [4 1 4]]\n",
            "\n",
            "  [[2 2 4]\n",
            "   [2 3 3]\n",
            "   [1 4 4]]\n",
            "\n",
            "  [[2 4 3]\n",
            "   [3 3 0]\n",
            "   [4 4 0]]\n",
            "\n",
            "  [[4 3 4]\n",
            "   [3 0 2]\n",
            "   [4 0 2]]\n",
            "\n",
            "  [[3 4 4]\n",
            "   [0 2 3]\n",
            "   [0 2 2]]\n",
            "\n",
            "  [[4 4 2]\n",
            "   [2 3 4]\n",
            "   [2 2 4]]\n",
            "\n",
            "  [[4 2 0]\n",
            "   [3 4 1]\n",
            "   [2 4 3]]]\n",
            "\n",
            "\n",
            " [[[1 3 2]\n",
            "   [3 4 1]\n",
            "   [0 4 0]]\n",
            "\n",
            "  [[3 2 3]\n",
            "   [4 1 4]\n",
            "   [4 0 4]]\n",
            "\n",
            "  [[2 3 3]\n",
            "   [1 4 4]\n",
            "   [0 4 3]]\n",
            "\n",
            "  [[3 3 0]\n",
            "   [4 4 0]\n",
            "   [4 3 2]]\n",
            "\n",
            "  [[3 0 2]\n",
            "   [4 0 2]\n",
            "   [3 2 3]]\n",
            "\n",
            "  [[0 2 3]\n",
            "   [0 2 2]\n",
            "   [2 3 0]]\n",
            "\n",
            "  [[2 3 4]\n",
            "   [2 2 4]\n",
            "   [3 0 4]]\n",
            "\n",
            "  [[3 4 1]\n",
            "   [2 4 3]\n",
            "   [0 4 0]]]]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "Z = np.random.randint(0,5,(10,10))\n",
        "n = 3\n",
        "i = 1 + (Z.shape[0]-3)\n",
        "j = 1 + (Z.shape[1]-3)\n",
        "c = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)\n",
        "print(c)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "q1__rzuGqSfI"
      },
      "source": [
        "#### 85. Create a 2D array subclass such that Z\\[i,j\\] == Z\\[j,i\\] (★★★) \n",
        "(**hint**: class method)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 505,
      "metadata": {
        "id": "uhzmXXcsqSfJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8a2dd372-14f8-42f8-cfb1-1fe93846aae8"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 2  7 11 15  6]\n",
            " [ 7  7  8  7  7]\n",
            " [11  8  9 42 16]\n",
            " [15  7 42  9 11]\n",
            " [ 6  7 16 11  4]]\n"
          ]
        }
      ],
      "source": [
        "class Symetric(np.ndarray):\n",
        "    def __setitem__(self, index, value):\n",
        "        i,j = index\n",
        "        super(Symetric, self).__setitem__((i,j), value)\n",
        "        super(Symetric, self).__setitem__((j,i), value)\n",
        "\n",
        "def symetric(Z):\n",
        "    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)\n",
        "\n",
        "S = symetric(np.random.randint(0,10,(5,5)))\n",
        "S[2,3] = 42\n",
        "print(S)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VJBFgCipqSfJ"
      },
      "source": [
        "#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) \n",
        "(**hint**: np.tensordot)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 506,
      "metadata": {
        "id": "MlRiavGLqSfK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ee013895-12b1-47a3-cde8-ad9733b2d210"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]\n",
            " [200.]]\n"
          ]
        }
      ],
      "source": [
        "p, n = 10, 20\n",
        "M = np.ones((p,n,n))\n",
        "V = np.ones((p,n,1))\n",
        "S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])\n",
        "print(S)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oIuCm9vpqSfK"
      },
      "source": [
        "#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) \n",
        "(**hint**: np.add.reduceat)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 507,
      "metadata": {
        "id": "W4IL00BMqSfL",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d72fddcc-2894-4fe2-cf2c-68e008b6cadd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[16. 16. 16. 16.]\n",
            " [16. 16. 16. 16.]\n",
            " [16. 16. 16. 16.]\n",
            " [16. 16. 16. 16.]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.ones((16,16))\n",
        "k = 4\n",
        "S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),\n",
        "                                       np.arange(0, Z.shape[1], k), axis=1)\n",
        "print(S)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ssPcuDT9qSfL"
      },
      "source": [
        "#### 88. How to implement the Game of Life using numpy arrays? (★★★)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 508,
      "metadata": {
        "id": "bVYxCk-SqSfM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f7149e89-7827-44c3-d055-379d7ba0c469"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0\n",
            "  0 0 1 1 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 0\n",
            "  0 1 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0\n",
            "  0 1 0 0 1 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 1 0 0 0 1 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 1 1 0 1 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0\n",
            "  1 0 0 0 0 1 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 0 0\n",
            "  1 0 0 0 0 1 0 0 0 0 1 1 0 0]\n",
            " [0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  1 1 0 1 1 0 0 0 0 0 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 1 0]\n",
            " [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 1 1 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 0 1 1 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n",
            "  0 0 0 1 0 0 0 0 1 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n",
            "  0 0 1 1 0 0 0 0 1 0 0 1 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 1 1 0 0]\n",
            " [0 0 0 0 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 0 0 0 0]\n",
            " [0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 1 0 0 0 0]\n",
            " [0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 1 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 0 1 1 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 1 0 0 1 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 0 1 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 1 1 1 1 0]\n",
            " [0 1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 1 1 0 1 0]\n",
            " [0 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 1 0 0 0]\n",
            " [0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 1 1 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0\n",
            "  0 0 1 1 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0\n",
            "  0 0 1 1 0 0 0 0 0 1 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 1 0 0 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 1 0 0 0 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 1 1 0 1 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 1 1 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 1 1 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 1 1 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 1 1 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
            "  0 0 0 0 0 0 0 0 0 0 0 0 0 0]]\n"
          ]
        }
      ],
      "source": [
        "def iterate(Z):\n",
        "    # Count neighbours\n",
        "    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +\n",
        "         Z[1:-1,0:-2]                + Z[1:-1,2:] +\n",
        "         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])\n",
        "\n",
        "    # Apply rules\n",
        "    birth = (N==3) & (Z[1:-1,1:-1]==0)\n",
        "    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)\n",
        "    Z[...] = 0\n",
        "    Z[1:-1,1:-1][birth | survive] = 1\n",
        "    return Z\n",
        "\n",
        "Z = np.random.randint(0,2,(50,50))\n",
        "for i in range(100): Z = iterate(Z)\n",
        "print(Z)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zI-qnDGIqSfM"
      },
      "source": [
        "#### 89. How to get the n largest values of an array (★★★) \n",
        "(**hint**: np.argsort | np.argpartition)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 509,
      "metadata": {
        "id": "37lRlaDcqSfM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "df69d867-cd55-4aa6-be39-2cb50b32a233"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[9995 9996 9997 9998 9999]\n",
            "[9997 9999 9998 9996 9995]\n"
          ]
        }
      ],
      "source": [
        "Z = np.arange(10000)\n",
        "np.random.shuffle(Z)\n",
        "n = 5\n",
        "\n",
        "# Slow\n",
        "print (Z[np.argsort(Z)[-n:]])\n",
        "\n",
        "# Fast\n",
        "print (Z[np.argpartition(-Z,n)[:n]])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gq7BkLoPqSfN"
      },
      "source": [
        "#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) \n",
        "(**hint**: np.indices)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 510,
      "metadata": {
        "scrolled": true,
        "id": "2qhdaPBgqSfN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "36e5170e-a00c-47bd-caba-b2e802c05df4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1 4 6]\n",
            " [1 4 7]\n",
            " [1 5 6]\n",
            " [1 5 7]\n",
            " [2 4 6]\n",
            " [2 4 7]\n",
            " [2 5 6]\n",
            " [2 5 7]\n",
            " [3 4 6]\n",
            " [3 4 7]\n",
            " [3 5 6]\n",
            " [3 5 7]]\n"
          ]
        }
      ],
      "source": [
        "def cartesian(arrays):\n",
        "    arrays = [np.asarray(a) for a in arrays]\n",
        "    shape = (len(x) for x in arrays)\n",
        "\n",
        "    ix = np.indices(shape, dtype=int)\n",
        "    ix = ix.reshape(len(arrays), -1).T\n",
        "\n",
        "    for n, arr in enumerate(arrays):\n",
        "        ix[:, n] = arrays[n][ix[:, n]]\n",
        "\n",
        "    return ix\n",
        "\n",
        "print (cartesian(([1, 2, 3], [4, 5], [6, 7])))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "geplOOjRqSfN"
      },
      "source": [
        "#### 91. How to create a record array from a regular array? (★★★) \n",
        "(**hint**: np.core.records.fromarrays)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 511,
      "metadata": {
        "id": "6RllET0FqSfO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "706c93d1-3574-4b5d-a4f4-ca801c0fce5b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[(b'Hello', 2.5, 3) (b'World', 3.6, 2)]\n"
          ]
        }
      ],
      "source": [
        "Z = np.array([(\"Hello\", 2.5, 3),\n",
        "              (\"World\", 3.6, 2)])\n",
        "R = np.core.records.fromarrays(Z.T,\n",
        "                               names='col1, col2, col3',\n",
        "                               formats = 'S8, f8, i8')\n",
        "print(R)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ED2WPdNQqSfO"
      },
      "source": [
        "#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) \n",
        "(**hint**: np.power, \\*, np.einsum)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 512,
      "metadata": {
        "id": "v_elm5Q0qSfO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "406047e8-76e9-4287-f86a-e02f81bfb304"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1 loop, best of 5: 3.81 s per loop\n",
            "10 loops, best of 5: 160 ms per loop\n",
            "10 loops, best of 5: 143 ms per loop\n"
          ]
        }
      ],
      "source": [
        "x = np.random.rand(int(5e7))\n",
        "\n",
        "%timeit np.power(x,3)\n",
        "%timeit x*x*x\n",
        "%timeit np.einsum('i,i,i->i',x,x,x)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ly5Lg28MqSfO"
      },
      "source": [
        "#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) \n",
        "(**hint**: np.where)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 513,
      "metadata": {
        "id": "bDggzdzMqSfO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4d4f48dc-1abf-4ba8-db05-8ca1fc214944"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 1 2 4 5 6]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "A = np.random.randint(0,5,(8,3))\n",
        "B = np.random.randint(0,5,(2,2))\n",
        "\n",
        "C = (A[..., np.newaxis, np.newaxis] == B)\n",
        "rows = np.where(C.any((3,1)).all(1))[0]\n",
        "print(rows)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9SgvYbHGqSfP"
      },
      "source": [
        "#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \\[2,2,3\\]) (★★★)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 514,
      "metadata": {
        "id": "Qh-j90YAqSfP",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1af88d48-5370-4541-872f-66df78b6b665"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 2 2]\n",
            " [1 4 4]\n",
            " [4 3 1]\n",
            " [4 1 1]\n",
            " [4 4 3]\n",
            " [1 2 4]\n",
            " [2 2 0]\n",
            " [0 0 4]\n",
            " [3 3 3]\n",
            " [4 3 4]]\n",
            "[[0 2 2]\n",
            " [1 4 4]\n",
            " [4 3 1]\n",
            " [4 1 1]\n",
            " [4 4 3]\n",
            " [1 2 4]\n",
            " [2 2 0]\n",
            " [0 0 4]\n",
            " [4 3 4]]\n",
            "[[0 2 2]\n",
            " [1 4 4]\n",
            " [4 3 1]\n",
            " [4 1 1]\n",
            " [4 4 3]\n",
            " [1 2 4]\n",
            " [2 2 0]\n",
            " [0 0 4]\n",
            " [4 3 4]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.randint(0,5,(10,3))\n",
        "print(Z)\n",
        "# solution for arrays of all dtypes (including string arrays and record arrays)\n",
        "E = np.all(Z[:,1:] == Z[:,:-1], axis=1)\n",
        "U = Z[~E]\n",
        "print(U)\n",
        "# soluiton for numerical arrays only, will work for any number of columns in Z\n",
        "U = Z[Z.max(axis=1) != Z.min(axis=1),:]\n",
        "print(U)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hwlmN43NqSfP"
      },
      "source": [
        "#### 95. Convert a vector of ints into a matrix binary representation (★★★) \n",
        "(**hint**: np.unpackbits)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 515,
      "metadata": {
        "id": "Zz-YzXvNqSfP",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c0b39d10-342b-4eaa-a4ca-ea7c92ed0e70"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 0 0 0 0 0 0 0]\n",
            " [0 0 0 0 0 0 0 1]\n",
            " [0 0 0 0 0 0 1 0]\n",
            " [0 0 0 0 0 0 1 1]\n",
            " [0 0 0 0 1 1 1 1]\n",
            " [0 0 0 1 0 0 0 0]\n",
            " [0 0 1 0 0 0 0 0]\n",
            " [0 1 0 0 0 0 0 0]\n",
            " [1 0 0 0 0 0 0 0]]\n"
          ]
        }
      ],
      "source": [
        "I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])\n",
        "B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)\n",
        "print(B[:,::-1])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "F_1lYWiJqSfP"
      },
      "source": [
        "#### 96. Given a two dimensional array, how to extract unique rows? (★★★) \n",
        "(**hint**: np.ascontiguousarray)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 516,
      "metadata": {
        "id": "fAjGgnD-qSfQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7a8fbf57-74ff-4e69-b50d-ae35b1d688df"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0 0 0]\n",
            " [0 0 1]\n",
            " [0 1 1]\n",
            " [1 0 1]\n",
            " [1 1 1]]\n"
          ]
        }
      ],
      "source": [
        "Z = np.random.randint(0,2,(6,3))\n",
        "T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))\n",
        "_, idx = np.unique(T, return_index=True)\n",
        "uZ = Z[idx]\n",
        "print(uZ)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iLiD6UldqSfQ"
      },
      "source": [
        "#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) \n",
        "(**hint**: np.einsum)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 517,
      "metadata": {
        "id": "dHXUyIb5qSfQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8aa2866c-0b98-47c1-ed9c-9eeb5bff3efa"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.14214652, 0.70580604, 0.30660006, 0.42858609, 0.64694077,\n",
              "        0.38109082, 0.37416963, 0.53630655, 0.08248248, 0.16740915],\n",
              "       [0.10326869, 0.51276434, 0.22274331, 0.31136552, 0.46999903,\n",
              "        0.27686045, 0.27183225, 0.38962386, 0.05992309, 0.12162186],\n",
              "       [0.03738407, 0.18562467, 0.08063481, 0.11271673, 0.1701433 ,\n",
              "        0.10022563, 0.09840538, 0.14104686, 0.02169262, 0.04402806],\n",
              "       [0.07097658, 0.35242295, 0.15309149, 0.21400153, 0.32303035,\n",
              "        0.1902862 , 0.18683031, 0.2677885 , 0.04118514, 0.08359071],\n",
              "       [0.02425138, 0.12041638, 0.05230852, 0.07312035, 0.11037347,\n",
              "        0.06501726, 0.06383645, 0.09149836, 0.0140722 , 0.02856139],\n",
              "       [0.08816054, 0.43774723, 0.1901561 , 0.26581293, 0.40123846,\n",
              "        0.23635594, 0.23206335, 0.33262213, 0.05115637, 0.10382865],\n",
              "       [0.11480437, 0.57004293, 0.24762496, 0.34614676, 0.5225005 ,\n",
              "        0.30778728, 0.3021974 , 0.43314698, 0.06661682, 0.13520768],\n",
              "       [0.12869066, 0.63899312, 0.27757672, 0.38801533, 0.58570014,\n",
              "        0.34501605, 0.33875003, 0.48553877, 0.07467454, 0.15156188],\n",
              "       [0.04811641, 0.23891443, 0.10378372, 0.14507583, 0.21898861,\n",
              "        0.12899875, 0.12665593, 0.18153907, 0.02792021, 0.05666778],\n",
              "       [0.03082152, 0.15303936, 0.06647985, 0.09292998, 0.14027565,\n",
              "        0.08263162, 0.0811309 , 0.11628692, 0.01788461, 0.03629919]])"
            ]
          },
          "metadata": {},
          "execution_count": 517
        }
      ],
      "source": [
        "A = np.random.uniform(0,1,10)\n",
        "B = np.random.uniform(0,1,10)\n",
        "\n",
        "np.einsum('i->', A)       # np.sum(A)\n",
        "np.einsum('i,i->i', A, B) # A * B\n",
        "np.einsum('i,i', A, B)    # np.inner(A, B)\n",
        "np.einsum('i,j->ij', A, B)    # np.outer(A, B)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GL8qhK01qSfQ"
      },
      "source": [
        "#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? \n",
        "(**hint**: np.cumsum, np.interp)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 518,
      "metadata": {
        "collapsed": true,
        "id": "gOs6LAjPqSfR"
      },
      "outputs": [],
      "source": [
        "phi = np.arange(0, 10*np.pi, 0.1)\n",
        "a = 1\n",
        "x = a*phi*np.cos(phi)\n",
        "y = a*phi*np.sin(phi)\n",
        "\n",
        "dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths\n",
        "r = np.zeros_like(x)\n",
        "r[1:] = np.cumsum(dr)                # integrate path\n",
        "r_int = np.linspace(0, r.max(), 200) # regular spaced path\n",
        "x_int = np.interp(r_int, r, x)       # integrate path\n",
        "y_int = np.interp(r_int, r, y)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Jd_2SeWFqSfR"
      },
      "source": [
        "#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) \n",
        "(**hint**: np.logical\\_and.reduce, np.mod)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 519,
      "metadata": {
        "id": "QZQytAorqSfR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "59af866e-bff6-4274-fe9b-9ca08add079a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[2. 0. 1. 1.]]\n"
          ]
        }
      ],
      "source": [
        "X = np.asarray([[1.0, 0.0, 3.0, 8.0],\n",
        "                [2.0, 0.0, 1.0, 1.0],\n",
        "                [1.5, 2.5, 1.0, 0.0]])\n",
        "n = 4\n",
        "M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)\n",
        "M &= (X.sum(axis=-1) == n)\n",
        "print(X[M])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GEaiMvMUqSfS"
      },
      "source": [
        "#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) \n",
        "(**hint**: np.percentile)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 520,
      "metadata": {
        "id": "WoCy7fdPqSfS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ce139ab6-e101-4d04-bd5f-13821539518e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[-0.06229055  0.30320151]\n"
          ]
        }
      ],
      "source": [
        "X = np.random.randn(100) # random 1D array\n",
        "N = 1000 # number of bootstrap samples\n",
        "idx = np.random.randint(0, X.size, (N, X.size))\n",
        "means = X[idx].mean(axis=1)\n",
        "confint = np.percentile(means, [2.5, 97.5])\n",
        "print(confint)"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.3"
    },
    "colab": {
      "name": "Copy of Numpy_tasks.py",
      "provenance": [],
      "collapsed_sections": [
        "PhJtEmD7qSeY",
        "XNYONM9OqSeZ",
        "DEHVly65qSeb",
        "r1s2OvcDqSec",
        "CsgthtW1qSfH",
        "q1__rzuGqSfI",
        "VJBFgCipqSfJ",
        "oIuCm9vpqSfK",
        "ssPcuDT9qSfL",
        "zI-qnDGIqSfM",
        "gq7BkLoPqSfN",
        "geplOOjRqSfN",
        "ED2WPdNQqSfO",
        "9SgvYbHGqSfP",
        "hwlmN43NqSfP",
        "F_1lYWiJqSfP",
        "iLiD6UldqSfQ",
        "GL8qhK01qSfQ",
        "Jd_2SeWFqSfR",
        "GEaiMvMUqSfS"
      ],
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}