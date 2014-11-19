import System.Environment (getArgs)

data VSO = VSO { verb :: String, subject :: String, object :: String, instances :: Int } 
instance Eq VSO where 
  (VSO v s o _) == (VSO v' s' o' _) = v == v' && s == s' && o == o'
instance Show VSO where
  show (VSO v s o c) = v ++ " " ++ s ++ " " ++ o ++ " " ++ show c ++ " \n"

writeConcat :: String -> VSO -> [VSO] -> IO ()
writeConcat outF current [] = (appendFile outF) (show current)
writeConcat outF current (next:rest) 
  | next == current = writeConcat outF current' rest
  | otherwise       = do (appendFile outF) (show current)
                         writeConcat outF next rest
  where current' = VSO (verb current) 
                       (subject current) 
                       (object current) 
                       (instances current + instances next)

readVSO :: String -> VSO
readVSO st | length (words st) /= 4 = error $ "problem: " ++ st
           | otherwise = (VSO v s o (read c :: Int))
  where [v,s,o,c] = words st

main :: IO ()
main = do  
  [inF]  <- getArgs
  stream <- readFile inF
  let outF   = inF ++ ".concat"
  let vsos = map readVSO (lines stream)
  writeConcat outF (head vsos) (tail vsos)


