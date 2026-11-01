# R DESeq2와 남은 차이

기준: R DESeq2 1.50.2 (R 4.5.3), rust_deseq2 `fix/port-fidelity` 브랜치(커밋 `aec02e0` 시점), 2026-11-01.

R 쪽 목록은 `getNamespaceExports("DESeq2")`와 주요 함수의 `formals()`에서 뽑았고, Rust 쪽은 CLI 옵션
(`run`/`vst`/`rlog`/`normalize --help`)과 라이브러리 공개 함수에서 하나씩 대조했다. "구현됨"이라고 적은 것은
`validation/`의 R 대조 하네스로 확인한 기능만이다.

## 1. 요약

| 구분 | 내용 |
|---|---|
| 구현되어 R과 일치 | 표준 DESeq2 파이프라인 전체(size factor 3종, dispersion 3종, Wald/LRT, Cook's 이상치 대체·필터, 독립 필터링, `lfcThreshold`, `useT`, 관측 가중치, normalizationFactors), 수축 3종, VST(전체·`nsub`), rlog, 입력(행렬, htseq-count, tximport 7종 + kallisto h5) |
| 구현 안 된 분석 기능 | `betaPrior = TRUE`, `glmGamPoi`, `results()`의 다른 대립가설·다른 p 보정, 가중치 + normal/apeglm 수축 등 (2절) |
| CLI에서 고정되었거나 노출되지 않은 것 | VST/rlog의 size factor 방식, rlog의 추세 방식, `minmu`, `useCR`, `collapseReplicates`·`fpkm`·`fpm` 등 (3절) |
| 없는 보조 함수 | 그림 5종(`plotPCA` 등), `normTransform`, `unmix`, `integrateWithSingleCell`, `makeExampleDESeqDataSet` 등 (5절) |
| 구현됐지만 숫자가 미세하게 다른 곳 | poscounts·iterate size factor, 우도가 평평한 유전자, 수축 (6절). 모든 검증 시나리오에서 유의성·NA 판정 불일치는 0 |

## 2. 구현되지 않은 분석 기능

### `DESeq()` / `nbinomWaldTest()` / `nbinomLRT()`

| R 기능 | 상태 | 비고 |
|---|---|---|
| `betaPrior = TRUE` (`estimateBetaPriorVar`, `estimateMLEForBetaPriorVar`) | 미구현 | DESeq2 1.16 이전의 기본값. 지금 R 기본값은 `FALSE` |
| `modelMatrixType = "expanded"` | 미구현 | `betaPrior = TRUE`와 함께 쓰임 |
| `fitType = "glmGamPoi"`, `nbinomLRT(type = "glmGamPoi")` | 미구현 | glmGamPoi 패키지 백엔드(단일세포용). README 로드맵 3단계 |
| `full`/`reduced`에 모델 행렬 직접 지정, `modelMatrix =` | 미구현 | CLI는 식(formula)만 받음 |
| `minmu` 변경 | 미구현 | 0.5 고정(R 기본값과 같음) |
| `nbinomWaldTest(df =)` 사용자 지정 t 자유도 | 미노출 | 내부 함수에는 인자가 있으나 CLI 옵션 없음. `--use-t`는 R 기본 자유도를 씀 |
| `useOptim = FALSE`, `useQR = FALSE` | 미노출 | 항상 R 기본값(`TRUE`)으로 동작 |

### 설계식

| R 기능 | 상태 | 비고 |
|---|---|---|
| 임의 formula (`~ a + b + a:b`, `~ a * b`, `~ 0 + a` 등) | **구현됨** | `--formula`(`run`, `vst`, `rlog`), 라이브러리 `builder().formula()`. 연산자 `+ : * - / %in% ^`, 괄호, `0 +`/`- 1`. R `model.matrix()`의 열 순서·코딩(주효과가 없는 상호작용의 marginality 규칙 포함)과 DESeq2의 `resultsNames`를 따른다. 숫자 변수는 `--continuous`로 지정(R은 colData 열의 형으로 정함), 값이 모두 `TRUE`/`FALSE`인 열은 R처럼 logical(`flagTRUE`, 이름 바꿈 없음). `.`은 R `DESeqDataSet()`처럼 오류. `validation/gen12.R` 22개 시나리오 |
| formula 안의 함수 | **구현됨** | R `model.frame()`처럼 평가: `factor()`, `as.factor()`, `relevel()`, `ordered()`(`contr.poly` 다항 대비), `as.numeric()`/`as.integer()`, `log()`(`base`), `log2`, `log10`, `log1p`, `exp`, `expm1`, `sqrt`, `abs`, `scale()`, `I()`(사칙연산·`^`·`%%`·비교·`%in%`), `poly()`(직교: LINPACK `dqrdc2` QR, `raw = TRUE`), `splines::ns()`/`bs()`(`splineDesign()`, type 7 분위수 매듭, 자연 스플라인 QR 제약), `offset()`(DESeq2처럼 무시). 열 이름은 R `deparse()` + `make.names()`(`poly.x..2.1`), `_vs_` 이름 바꿈은 `all.vars(design)`의 colData factor만(`factor.dose.2`는 그대로). `--contrast-var`는 R처럼 colData factor여야 함. 모형 행렬 124개가 R과 4e-16까지 같음(`validation/gen_formula_cases.R`), `validation/gen13.R` 30개 시나리오. 미지원: 그 밖의 R 함수와 사용자 정의 함수(R 코드를 실행하지 않음, 오류 메시지에 지원 함수 목록), `polym`(`poly()`에 변수 여러 개), `poly(coefs =)`, `factor(exclude =, nmax =)`, `NA` |
| 상호작용 항 | **구현됨** | `--formula`로. 계수 이름은 R과 같은 `batchb2.conditionB`. builder의 `.interaction()`도 같은 경로를 쓴다(예전 이름 `f1_l1_x_f2_l2`와 dispersion에서 상호작용 열을 빼던 문제 수정). 상호작용이 있으면 normal 수축은 R처럼 오류 |
| 절편 없는 설계 (`~ 0 + a`) | **구현됨** | 열 이름을 바꾸지 않음(`conditionA`), `--numerator`/`--denominator`는 R `cleanContrast()`의 규칙(`noInterceptPullCoef` 포함), 기본 대비는 R처럼 `c(마지막 변수, 마지막 수준, 첫 수준)` |
| LRT 축소 모형 | **구현됨** | `--reduced`가 같은 파서를 쓰고 R `checkLRT()`(축소식 변수는 전체식에 있어야 함)와 자유도 검사. full rank가 아닌 축소 모형도 R `fitNbinomGLMs()`처럼 적합(절편은 `log(baseMean)`, 나머지 0에서 시작, 절편이 없으면 모두 1; ridge 벌점) |
| `DESeqDataSetFromMatrix(ignoreRank = TRUE)` | 미지원 | 계수 행렬이 full rank가 아니면 오류 |

### `results()`

| R 기능 | 상태 | 비고 |
|---|---|---|
| `name =` (계수 이름으로 결과 추출) | **구현됨** | `--name`. 연속 변수 기울기·공변량 계수. 대비 없이 실행하면 R처럼 마지막 계수. 공변량의 수준 대비는 `--contrast-var` |
| `contrast =` 숫자 벡터, `list(...)` + `listValues` | **구현됨** | `--contrast-vector`, `--contrast-list` + `--list-values`. LRT에서는 보고할 LFC 열을 정함 |
| `altHypothesis = "lessAbs"`, `"greater"`, `"less"`, `"greaterAbsUPSHOT"`, `"greaterAbs2014"` | 미지원 | `greaterAbs`(기본값)만 |
| `pAdjustMethod` (BH 외) | 미노출 | BH 고정. 라이브러리에 Bonferroni 함수는 있음 |
| `filter`, `filterFun`, `theta` (독립 필터링 사용자 지정) | 미지원 | R 기본 방식(baseMean 기준)만 |
| `addMLE = TRUE`, `saveCols` | 미지원 | |
| 한 번 적합한 모델에서 여러 대비 추출 | 미지원(CLI) | 대비마다 CLI를 다시 실행해야 함 |
| `format = "GRanges"`, `tidy` | 해당 없음 | R 객체 형식 |

### `lfcShrink()`

| R 기능 | 상태 | 비고 |
|---|---|---|
| 관측 가중치가 있을 때 `type = "normal"`, `"apeglm"` | 미구현 | 오류로 중단. `ashr`는 가능 |
| `svalue = TRUE` (s-value 출력) | 미구현 | ashr 내부에서 s-value를 계산하지만 출력하지 않음 |
| `lfcShrink(lfcThreshold =)` | 미구현 | |
| `coef =`로 공변량 등 다른 계수 수축 | **구현됨** | normal/apeglm이 `--name` 등 대비가 가리키는 계수를 수축 |
| `apeAdapt`, `apeMethod` 변경 | 미노출 | R 기본값으로 동작 |

### 기타

| R 기능 | 상태 | 비고 |
|---|---|---|
| `estimateSizeFactors(controlGenes =, geoMeans =, locfunc =)` | 미지원 | 대조 유전자 지정, 외부 기하평균, `shorth` 등 다른 위치 함수 |
| `estimateDispersions(useCR = FALSE)` | 미지원 | Cox-Reid 보정 항상 사용(R 기본값) |
| `estimateDispersions(weightThreshold =)` | 미노출 | 0.01 고정(R 기본값) |
| `replaceOutliers(whichSamples =)` | 미지원 | |
| 사용자 정의 dispersion 함수(`dispersionFunction(dds) <-`) | 부분 | 라이브러리 setter만 있고 CLI 없음 |

## 3. CLI에서 고정되었거나 노출되지 않은 것

| 항목 | 현재 | R |
|---|---|---|
| `vst`의 size factor | ratio 고정 | 객체에 있는 size factor/`sfType` 사용 가능 |
| `rlog`의 size factor, 추세 | ratio, parametric 고정 | `fitType` 선택 가능 |
| `rlog(intercept =, betaPriorVar =)` | 미지원 | |
| `vst`/`rlog`의 `blind` 기본값 | `--blind`를 줘야 `blind = TRUE` | 기본값 `TRUE` |
| VST/rlog에 관측 가중치 | 적용 안 함 | |
| `collapseReplicates` | 라이브러리 함수만 | |
| `fpkm(robust =)`, `fpm` | 라이브러리 함수만, 유전자 길이 입력 CLI 없음 | |

## 4. 입력 (tximport 쪽)

DESeq2의 `DESeqDataSetFromMatrix`/`HTSeqCount`/`Tximport` 세 생성자는 모두 구현됐다. 남은 것은 tximport
패키지 쪽 기능이다.

| 기능 | 상태 |
|---|---|
| alevin (단일세포) | 미지원 |
| 사용자 지정 열(`importer`, `txIdCol`, `abundanceCol` 등, `type = "none"`) | 미지원 (`-c --avg-tx-length`로 우회) |
| `countsFromAbundance = "scaledTPM"`, `"lengthScaledTPM"`, `"dtuScaledTPM"`을 파일에서 직접 계산 | 미지원 (R이 만든 count 행렬은 `-c --tximport`로 입력 가능) |
| `ignoreTxVersion`, `ignoreAfterBar`, `txOut = TRUE` | 미지원 |
| inferential replicates(bootstrap) 읽기, `varReduce`, `infRepStat` | 미지원 (count·길이에는 영향 없음) |
| readr 설치 시 tximport의 숫자 파싱 | 미대조 (`read.delim` 경로와는 비트 단위 동일) |

## 5. 없는 보조 함수

| R 함수 | 용도 |
|---|---|
| `plotPCA` (`returnData = TRUE`의 PCA 좌표 포함), `plotMA`, `plotDispEsts`, `plotCounts`, `plotSparsity` | 그림 |
| `normTransform` | log2(정규화 count + 1) |
| `normalizeGeneLength` | 유전자 길이 보정 |
| `unmix` | 혼합 샘플 분해 |
| `integrateWithSingleCell` | 단일세포 참조와 통합 |
| `makeExampleDESeqDataSet` | 예제 데이터 생성 |
| `removeResults`, `priorInfo` 등 객체 조작 | R 객체 기능 (CLI 구조상 대부분 해당 없음) |

`parallel`/`BPPARAM`은 rayon 병렬 처리(`-t, --threads`)로 대체되어 기능 차이가 아니다.

## 6. 구현됐지만 숫자가 미세하게 다른 곳

모두 유의성(padj < 0.1)·NA 판정 불일치는 0이다. 수치는 R 대비 최대 절대 차이.

| 경우 | 차이 | 원인 |
|---|---|---|
| `--sf-type poscounts` (합성 데이터) | LFC 1.6e-3 | 평평한 목적함수에서 OLS 시작값의 마지막 비트 차이 |
| `--sf-type iterate` | LFC 1.75e-4 | 반복 추정의 수렴 지점 |
| 실제 데이터 sau_poscounts | SE 3.6e-4 | 반올림 수준의 knife-edge. R 자체도 기준 수준만 바꿔도 같은 크기로 변함 |
| 우도가 평평한 유전자 1개 (kallisto/piscem 1.2e-5, kallisto h5 3.4e-6) | LFC | IRLS 수렴 지점. R도 normalization factor를 1e-12 흔들면 4.7e-6 움직임 |
| apeglm / ashr 수축 | 1e-6 ~ 2e-5 | 최적화 경로의 반올림 차이 |

그 밖의 모든 검증 시나리오(합성 30개 + 실제 GEO 데이터 18개 + 입력 형식 시나리오)는 출력 반올림 수준(약 5e-7)에서
일치한다. 세부 표는 `README.md`의 Accuracy 절에 있다.

## 7. 우선순위 제안

사용 빈도와 작업량을 기준으로:

1. ~~`results(name =)`와 숫자 대비~~: 구현됨 (`validation/gen11.R`, R과 5e-7~2.7e-6).
2. **임의 설계식(상호작용 포함)**: 설계 행렬 코드는 있으므로 식 파서와 CLI 노출, R 대조 검증이 필요하다.
3. **`altHypothesis`의 나머지와 `pAdjustMethod`**: 계산식이 단순하다.
4. **VST/rlog 옵션 정리**: size factor 방식과 rlog `fitType` 노출, `blind` 기본값을 R과 맞출지 결정.
5. **가중치 + normal/apeglm 수축**: R C++ 코드 이식이 필요해 작업량이 크다.
6. **`glmGamPoi`**: 단일세포 지원(README 로드맵)과 함께 진행할 큰 작업.
7. **그림, `normTransform` 등 보조 함수**: CLI 도구의 성격상 우선순위가 낮다(`plotPCA`의 좌표 출력 정도는 유용).
