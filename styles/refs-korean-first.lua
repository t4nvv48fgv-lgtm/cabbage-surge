-- refs-korean-first.lua — 참고문헌 국문 우선 재배열 (JRD: 국문 가나다 → 영문 알파벳)
-- citeproc가 생성한 #refs div의 항목을 한글 포함 여부로 안정 분할한다.
-- citeproc의 기존 정렬이 각 그룹 안에서 유지되므로 국문은 가나다, 영문은 알파벳 순이 보존된다.
-- 반드시 --citeproc "뒤에" 지정할 것: pandoc ... --citeproc --lua-filter=styles/refs-korean-first.lua

local function has_hangul(s)
  -- UTF-8에서 한글 음절(U+AC00~U+D7A3)의 선두 바이트는 0xEA~0xED 범위에만 나타난다.
  -- (연속 바이트는 0x80~0xBF라 오탐 없음; 본 서지에는 한글/라틴만 존재)
  return s:find("[\234-\237]") ~= nil
end

function Div(el)
  if el.identifier ~= "refs" then return nil end
  local kor, eng = {}, {}
  for _, item in ipairs(el.content) do
    if has_hangul(pandoc.utils.stringify(item)) then
      table.insert(kor, item)
    else
      table.insert(eng, item)
    end
  end
  for _, item in ipairs(eng) do table.insert(kor, item) end
  el.content = kor
  return el
end
