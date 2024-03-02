# Project 1 - MyAutoPano

# Phase 1: 



# Phase 2: Deep Learning Approach

- How to run :

    ```
    cd MyAutoPano\Phase2\Code
    ```
    - Make sure unzip all 3 data
        - Train.zip
        - Val.zip
        - P1TestSet.zip

    - Data Generate
    ```
    python generate_data.py
    ```

    - Train
    ```
    python Train.py --ModelType Sup
    ```
    - or
    ```
    python Train.py --ModelType UnSup
    ```

    - Test
    ```
    python test.py --ModelType {1} --TestSet test {2}
    ```
        - {1} should be Sup or UnSup
        - {2} should be test, train or val