#pragma once
#include <limits>
#include "ff_p.hpp"

inline constexpr uint64_t STATE_WIDTH = 12;
inline constexpr uint64_t RATE_WIDTH = 8;
inline constexpr uint64_t DIGEST_SIZE = 4;
inline constexpr uint64_t NUM_ROUNDS = 7;
inline constexpr uint64_t MAX_UINT = 0xFFFFFFFFULL;

/*
  Note : Actually I wanted to use `marray` instead of `vec`, but seems that
  `mul_hi` is not yet able to take `marray` as input in SYCL/ DPCPP

  I will come and take a look later !
*/

// Performs element wise modular multiplication on two operand vectors
//
// Returned vector may not have all elements in canonical representation
// so consider running `res % MOD` before consumption !
//
// Takes quite some motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L9-L36
__device__ ulong4
ff_p_vec_mul_(ulong4 a, ulong4 b);

__device__ void
ff_p_vec_mul(const ulong4* a,
             const ulong4* b,
             ulong4* const c);

// Performs element wise modular addition ( on aforementioned 64-bit prime field
// ) on two supplied operands
//
// Before consumption consider performing `res % MOD` so that all
// lanes are in canonical form
//
// Collects quite some motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L38-L66
__device__ ulong4
ff_p_vec_add_(const ulong4 &a, const ulong4 &b);

__device__ void
ff_p_vec_add(const ulong4* a,
             const ulong4* b,
             ulong4* const c);

// Updates each element of rescue prime hash state ( total 12 lane wide = 3 x 4
// -lane vectors ) by exponentiating to their 7-th power
//
// Note this implementation doesn't use modular exponentiation routine, instead
// it uses multiple multiplications ( actually squaring )
//
// Collects huge motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L68-L88
__device__ void
apply_sbox(const ulong4* state_in, ulong4* const state_out);

// Applies rescue round key constants on hash state
//
// actually simple vectorized modular addition --- that's all this routine does
//
// inline it ?
//
// Taken from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L97-L106
__device__ void
apply_constants(const ulong4* state_in,
                const ulong4* cnst,
                ulong4* const state_out);

// Reduces four prime field element vector into single accumulated prime
// field element, by performing modular addition
//
// Adapted from here
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L143-L166
__device__ ulong
accumulate_vec4(ulong4 a);

// Accumulates state of rescue prime hash into single prime field element
//
// Takes some inspiration from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L168-L199
__device__ ulong
accumulate_state(const ulong4* state);

// Performs matrix vector multiplication; updates state of rescue prime
// hash by applying MDS matrix
//
// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L201-L231
__device__ void
apply_mds(const ulong4* state_in,
          const ulong4* mds,
          ulong4* const state_out);

// Instead of exponentiating hash state by some large number, this function
// helps in computing exponentiation by performing multiple modular
// multiplications
//
// This is invoked from following `apply_inv_sbox` function ( multiple times )
// See
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L242-L258
// for source of inspiration
__device__ void
exp_acc(const ulong m,
        const ulong4* base,
        const ulong4* tail,
        ulong4* const out);

// Applies inverse sbox on rescue prime hash state --- this function
// actually exponentiates rescue prime hash state by [large
// number](https://github.com/itzmeanjan/ff-gpu/blob/a1d99fd84c221f90eafd050df57315b77cb425d7/include/test_rescue_prime.hpp#L7),
// but as an optimization step, instead of performing all these expensive
// modular exponentiations, 72 multiplications are performed
//
// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L260-L287
__device__ void
apply_inv_sbox(const ulong4* state_in, ulong4* const state_out);

// Apply a round of rescue permutation, which mixes/ consumes input into hash
// state
//
// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L296-L313
__device__ void
apply_permutation_round(const ulong4* state_in,
                        const ulong4* mds,
                        const ulong4* ark1,
                        const ulong4* ark2,
                        ulong4* const state_out);

// Applies all rounds ( = 7 ) of rescue permutation, updating hash state
//
// Once RATE_WIDTH -many field elements are consumed from input stream
// permutation is applied on hash state for mixing them well into state
//
// After all input is consumed, if there're some input which were
// not mixed well ( i.e. some input were read after last rescue
// permutation call ) this function will be required to be invoked again
//
// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L315-L332
__device__ void
apply_rescue_permutation(const ulong4* state_in,
                         const ulong4* mds,
                         const ulong4* ark1,
                         const ulong4* ark2,
                         ulong4* const state_out);

// Computes rescue prime hash of input prime field elements, by consuming
// all input elements into 12 elements wide hash state
//
// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500dd1f271e4f8badf1305c8077e2532eb510/kernel.cl#L345-L422
__device__ void
hash_elements(const ulong* input_elements,
              const ulong count,
              ulong* const hash,
              const ulong4* mds,
              const ulong4* ark1,
              const ulong4* ark2);


// Merges two rescue prime digests into single digest of width 256 -bit
//
// `input_hashes` parameter should be pointer to memory location, where 8
// consequtive prime field elements can be found, 8 because two rescue prime
// hashes contiguously stored
//
// `merged_hash` parameter should be pointing to 256-bit wide memory allocation
// where four field elements to be written
//
// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/77e371ef2fb11ba7d7369005a60a0888393729f0/kernel.cl#L424-L474
__device__ void
merge(const ulong* input_hashes,
      ulong* const merged_hash,
      const ulong4* mds,
      const ulong4* ark1,
      const ulong4* ark2);

// Stores MDS matrix in kernel expected form i.e. each row of matrix inside
// vector with 16 lanes
void
prepare_mds(ulong4* const mds);

// Stores round key constants matrix in kernel expected form i.e. each row of
// matrix inside vector with 16 lanes
void
prepare_ark1(ulong4* const ark1);

// Stores round key constants matrix in kernel expected form i.e. each row of
// matrix inside vector with 16 lanes
void
prepare_ark2(ulong4* const ark2);

inline constexpr ulong MDS[144] = {
  2108866337646019936ull,  11223275256334781131ull, 2318414738826783588ull,
  11240468238955543594ull, 8007389560317667115ull,  11080831380224887131ull,
  3922954383102346493ull,  17194066286743901609ull, 152620255842323114ull,
  7203302445933022224ull,  17781531460838764471ull, 2306881200ull,

  3368836954250922620ull,  5531382716338105518ull,  7747104620279034727ull,
  14164487169476525880ull, 4653455932372793639ul,   5504123103633670518ull,
  3376629427948045767ull,  1687083899297674997ull,  8324288417826065247ull,
  17651364087632826504ull, 15568475755679636039ull, 4656488262337620150ull,

  2560535215714666606ull,  10793518538122219186ull, 408467828146985886ull,
  13894393744319723897ull, 17856013635663093677ull, 14510101432365346218ull,
  12175743201430386993ull, 12012700097100374591ull, 976880602086740182ull,
  3187015135043748111ull,  4630899319883688283ull,  17674195666610532297ull,

  10940635879119829731ull, 9126204055164541072ull,  13441880452578323624ull,
  13828699194559433302ull, 6245685172712904082ull,  3117562785727957263ull,
  17389107632996288753ull, 3643151412418457029ull,  10484080975961167028ull,
  4066673631745731889ull,  8847974898748751041ull,  9548808324754121113ull,

  15656099696515372126ull, 309741777966979967ull,   16075523529922094036ull,
  5384192144218250710ull,  15171244241641106028ull, 6660319859038124593ull,
  6595450094003204814ull,  15330207556174961057ull, 2687301105226976975ull,
  15907414358067140389ull, 2767130804164179683ull,  8135839249549115549ull,

  14687393836444508153ull, 8122848807512458890ull,  16998154830503301252ull,
  2904046703764323264ull,  11170142989407566484ull, 5448553946207765015ull,
  9766047029091333225ull,  3852354853341479440ull,  14577128274897891003ull,
  11994931371916133447ull, 8299269445020599466ull,  2859592328380146288ull,

  4920761474064525703ull,  13379538658122003618ull, 3169184545474588182ull,
  15753261541491539618ull, 622292315133191494ull,   14052907820095169428ull,
  5159844729950547044ull,  17439978194716087321ull, 9945483003842285313ull,
  13647273880020281344ull, 14750994260825376ull,    12575187259316461486ull,

  3371852905554824605ull,  8886257005679683950ull,  15677115160380392279ull,
  13242906482047961505ull, 12149996307978507817ull, 1427861135554592284ull,
  4033726302273030373ull,  14761176804905342155ull, 11465247508084706095ull,
  12112647677590318112ull, 17343938135425110721ull, 14654483060427620352ull,

  5421794552262605237ull,  14201164512563303484ull, 5290621264363227639ull,
  1020180205893205576ull,  14311345105258400438ull, 7828111500457301560ull,
  9436759291445548340ull,  5716067521736967068ull,  15357555109169671716ull,
  4131452666376493252ull,  16785275933585465720ull, 11180136753375315897ull,

  10451661389735482801ull, 12128852772276583847ull, 10630876800354432923ull,
  6884824371838330777ull,  16413552665026570512ull, 13637837753341196082ull,
  2558124068257217718ull,  4327919242598628564ull,  4236040195908057312ull,
  2081029262044280559ull,  2047510589162918469ull,  6835491236529222042ull,

  5675273097893923172ull,  8120839782755215647ull,  9856415804450870143ull,
  1960632704307471239ull,  15279057263127523057ull, 17999325337309257121ull,
  72970456904683065ull,    8899624805082057509ull,  16980481565524365258ull,
  6412696708929498357ull,  13917768671775544479ull, 5505378218427096880ull,

  10318314766641004576ull, 17320192463105632563ull, 11540812969169097044ull,
  7270556942018024148ull,  4755326086930560682ull,  2193604418377108959ull,
  11681945506511803967ull, 8000243866012209465ull,  6746478642521594042ull,
  12096331252283646217ull, 13208137848575217268ull, 5548519654341606996ull,
};

inline constexpr ulong ARK1[84] = {
  13917550007135091859ull, 16002276252647722320ull, 4729924423368391595ull,
  10059693067827680263ull, 9804807372516189948ull,  15666751576116384237ull,
  10150587679474953119ull, 13627942357577414247ull, 2323786301545403792ull,
  615170742765998613ull,   8870655212817778103ull,  10534167191270683080ull,

  14572151513649018290ull, 9445470642301863087ull,  6565801926598404534ull,
  12667566692985038975ull, 7193782419267459720ull,  11874811971940314298ull,
  17906868010477466257ull, 1237247437760523561ull,  6829882458376718831ull,
  2140011966759485221ull,  1624379354686052121ull,  50954653459374206ull,

  16288075653722020941ull, 13294924199301620952ull, 13370596140726871456ull,
  611533288599636281ull,   12865221627554828747ull, 12269498015480242943ull,
  8230863118714645896ull,  13466591048726906480ull, 10176988631229240256ull,
  14951460136371189405ull, 5882405912332577353ull,  18125144098115032453ull,

  6076976409066920174ull,  7466617867456719866ull,  5509452692963105675ull,
  14692460717212261752ull, 12980373618703329746ull, 1361187191725412610ull,
  6093955025012408881ull,  5110883082899748359ull,  8578179704817414083ull,
  9311749071195681469ull,  16965242536774914613ull, 5747454353875601040ull,

  13684212076160345083ull, 19445754899749561ull,    16618768069125744845ull,
  278225951958825090ull,   4997246680116830377ull,  782614868534172852ull,
  16423767594935000044ull, 9990984633405879434ull,  16757120847103156641ull,
  2103861168279461168ull,  16018697163142305052ull, 6479823382130993799ull,

  13957683526597936825ull, 9702819874074407511ull,  18357323897135139931ull,
  3029452444431245019ull,  1809322684009991117ull,  12459356450895788575ull,
  11985094908667810946ull, 12868806590346066108ull, 7872185587893926881ull,
  10694372443883124306ull, 8644995046789277522ull,  1422920069067375692ull,

  17619517835351328008ull, 6173683530634627901ull,  15061027706054897896ull,
  4503753322633415655ull,  11538516425871008333ull, 12777459872202073891ull,
  17842814708228807409ull, 13441695826912633916ull, 5950710620243434509ull,
  17040450522225825296ull, 8787650312632423701ull,  7431110942091427450ull,
};

inline constexpr ulong ARK2[84] = {
  7989257206380839449ull,  8639509123020237648ull,  6488561830509603695ull,
  5519169995467998761ull,  2972173318556248829ull,  14899875358187389787ull,
  14160104549881494022ull, 5969738169680657501ull,  5116050734813646528ull,
  12120002089437618419ull, 17404470791907152876ull, 2718166276419445724ull,
  2485377440770793394ull,  14358936485713564605ull, 3327012975585973824ull,
  6001912612374303716ull,  17419159457659073951ull, 11810720562576658327ull,
  14802512641816370470ull, 751963320628219432ull,   9410455736958787393ull,
  16405548341306967018ull, 6867376949398252373ull,  13982182448213113532ull,
  10436926105997283389ull, 13237521312283579132ull, 668335841375552722ull,
  2385521647573044240ull,  3874694023045931809ull,  12952434030222726182ull,
  1972984540857058687ull,  14000313505684510403ull, 976377933822676506ull,
  8407002393718726702ull,  338785660775650958ull,   4208211193539481671ull,
  2284392243703840734ull,  4500504737691218932ull,  3976085877224857941ull,
  2603294837319327956ull,  5760259105023371034ull,  2911579958858769248ull,
  18415938932239013434ull, 7063156700464743997ull,  16626114991069403630ull,
  163485390956217960ull,   11596043559919659130ull, 2976841507452846995ull,
  15090073748392700862ull, 3496786927732034743ull,  8646735362535504000ull,
  2460088694130347125ull,  3944675034557577794ull,  14781700518249159275ull,
  2857749437648203959ull,  8505429584078195973ull,  18008150643764164736ull,
  720176627102578275ull,   7038653538629322181ull,  8849746187975356582ull,
  17427790390280348710ull, 1159544160012040055ull,  17946663256456930598ull,
  6338793524502945410ull,  17715539080731926288ull, 4208940652334891422ull,
  12386490721239135719ull, 10010817080957769535ull, 5566101162185411405ull,
  12520146553271266365ull, 4972547404153988943ull,  5597076522138709717ull,
  18338863478027005376ull, 115128380230345639ull,   4427489889653730058ull,
  10890727269603281956ull, 7094492770210294530ull,  7345573238864544283ull,
  6834103517673002336ull,  14002814950696095900ull, 15939230865809555943ull,
  12717309295554119359ull, 4130723396860574906ull,  7706153020203677238ull,
};
