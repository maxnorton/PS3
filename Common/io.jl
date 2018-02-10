module IO

export chkdir, fopen

function chkdir(path)
    if !isdir(path); mkdir(path) end
end

function fopen(filename)
    if is_windows()
        run(`$(ENV["COMSPEC"]) /c start $(filename)`)
    elseif is_apple()
        run(`open $(filename)`)
    elseif is_linux() || is_bsd()
        run(`xdg-open $(filename)`)
    else
        error("Unsupported platform.")
    end
end

end