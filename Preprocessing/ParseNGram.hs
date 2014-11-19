import Text.ParserCombinators.Parsec hiding (try)
import System.Environment (getArgs)
import Control.Monad (liftM, liftM3, join, msum)
import Data.List (find, elemIndex, delete)
import Data.Maybe (catMaybes, isJust, isNothing, fromJust)
import Data.Either (partitionEithers)
import Data.Time (getCurrentTime)
import Control.Exception (try, SomeException)

data NGramToken = NGramToken { word :: String, pos :: String, dep :: String, depHead :: Int } deriving Eq
type NGram = [NGramToken]

-- Takes an n-gram and returns Just the VSO part if has one; otherwise Nothing
pruneToVSO :: (NGram, Integer) -> Maybe (NGram, Integer)
pruneToVSO (ng, n) | length ng' < 3 = Nothing
                   | otherwise      = Just (ng', n)
  where ng' = catMaybes [v, s, o]
        v = find (\x -> pos x == "VB") ng
        vIndex = if isJust v then (fromJust $ elemIndex (fromJust v) ng) + 1 else 0
        s = find (\x -> if isJust v then dep x == "nsubj" && depHead x == vIndex else False) ng
        o = find (\x -> if isJust v then dep x == "dobj"  && depHead x == vIndex else False) ng

-- Parsers for the Google n-gram file format: (as specified in the README) 
-- http://storage.googleapis.com/books/syntactic-ngrams/index.html
nGramLine :: Parser (NGram, Integer)
nGramLine =
  do manyTill anyToken (char '\t') -- head-word (ignore)
     nGram <- sepBy1 nGramToken (char ' ') -- the ngram itself
     char '\t'
     c <- field -- total count
     many anyToken -- counts by date (ignore)
     return (nGram, (read c :: Integer))

field = many (noneOf [' ', '\t', '/'])

-- the word field is super annoying because it's allowed to have slashes in it.
-- Since it's the first to appear in the n-gram token, we have to count how many
-- slashes are in the token and then look for n-3 of them to be in the word
nGramToken :: Parser NGramToken
nGramToken =
  do nSlashes <- fmap (\x -> length . filter (=='/') $ x) (lookAhead $ many (noneOf [' ', '\t']))
     w <- fmap concat $ map sequence (iterate (++[(string "/"), field]) [field])!!(nSlashes-3)
     if length w == 0 then error $ "\nPROBLEM" ++ w else do
       char '/'
       (posTag, depTag, depHeadIndex) <- nGramTokenFields
       return (NGramToken w posTag depTag depHeadIndex)
 
nGramTokenFields = 
  do [posTag, depTag, depHeadStr] <- field `sepEndBy` (char '/') 
     let depHeadIndex = read depHeadStr :: Int
     return (posTag, depTag, depHeadIndex)

-- Format of the lines in our pre-processed VSO files 
-- verb<SPACE>subject<SPACE>object<SPACE>count<NEWLINE>
outFormatLine :: (NGram, Integer) -> String
outFormatLine ([v,s,o], c) = (word v) ++ (' ':(word s)) ++ (' ':(word o)) ++ (' ':(show c)) ++ "\n"

main :: IO ()
main = do 
  [inF] <- getArgs
  processFile inF

logLn :: String -> IO ()
logLn s = do
  time <- fmap show getCurrentTime
  let s' = time ++ "\t" ++ s ++ "\n"
  appendFile "ParseLog" s'
  putStrLn $ s

processFile :: String -> IO ()
processFile inF = do
  let outF = inF ++ ".prep"
  logLn $ "Processing file: " ++ inF 
  result <- (try $ readFile inF) :: IO (Either SomeException String)
  case result of 
    Left ex -> logLn $ "File " ++ inF ++ " not parsed: " ++ show ex ++ "\n"
    Right s -> do writeFile outF ""
                  processStream 0 0 outF (lines s)

processString :: Int -> String -> String -> IO Bool 
processString nScanned outF line = do
  either showError (processString' . pruneToVSO) (parse nGramLine line line)
  where
    showError e = do { logLn ("Parse Error on line " ++ show nScanned ++ show e ++ " ") ; return False }
    processString' l | isNothing l = do return False
                     | otherwise = do { ((appendFile outF) . outFormatLine) (fromJust l) ; return True}

processStream :: Int -> Int -> String -> [String] -> IO ()
processStream nScanned nVSOs outF [] = do 
  logLn $ "Wrote " ++ show nVSOs ++ " VSOs of " ++ show nScanned ++ " n-grams to: " ++ outF
processStream nScanned nVSOs outF (x:xs) = do 
  svo <- processString nScanned outF x 
  let nVSOs' = if svo then nVSOs + 1 else nVSOs 
  putStr $ "Scanned " ++ (show (nScanned+1)) ++ ". Found " ++ show nVSOs' ++ " VSO's.\r"
  processStream (nScanned+1) nVSOs' outF xs
